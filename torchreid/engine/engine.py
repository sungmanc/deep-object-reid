# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=too-many-branches,multiple-statements

from __future__ import absolute_import, division, print_function
import abc
import datetime
import math
import os
import os.path as osp
import time
from collections import namedtuple, OrderedDict
from copy import deepcopy
from torchreid.utils.tools import StateCacher, set_random_seed
import optuna

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR

from torchreid.integration.nncf.compression import (get_nncf_complession_stage,
                                                    get_nncf_prepare_for_tensorboard)
from torchreid.optim import ReduceLROnPlateauV2, WarmupScheduler, CosineAnnealingCycleRestart
from torchreid.utils import (AverageMeter, MetricMeter, get_model_attr,
                             open_all_layers, open_specified_layers,
                             save_checkpoint, ModelEmaV2, sample_mask)


EpochIntervalToValue = namedtuple('EpochIntervalToValue', ['first', 'last', 'value_inside', 'value_outside'])

def _get_cur_action_from_epoch_interval(epoch_interval, epoch):
    assert isinstance(epoch_interval, EpochIntervalToValue)
    if epoch_interval.first is None and epoch_interval.last is None:
        raise RuntimeError(f'Wrong epoch_interval {epoch_interval}')

    if epoch_interval.first is not None and epoch < epoch_interval.first:
        return epoch_interval.value_outside
    if epoch_interval.last is not None and epoch > epoch_interval.last:
        return epoch_interval.value_outside

    return epoch_interval.value_inside


class Engine(metaclass=abc.ABCMeta):
    r"""A generic base Engine class for both image- and video-reid."""
    def __init__(self,
                 datamanager,
                 models,
                 optimizers,
                 schedulers,
                 use_gpu=True,
                 save_all_chkpts=True,
                 train_patience = 10,
                 lr_decay_factor = 1000,
                 lr_finder = None,
                 early_stopping=False,
                 should_freeze_aux_models=False,
                 nncf_metainfo=None,
                 compression_ctrl=None,
                 initial_lr=None,
                 target_metric = 'train_loss',
                 epoch_interval_for_aux_model_freeze=None,
                 epoch_interval_for_turn_off_mutual_learning=None,
                 use_ema_decay=False,
                 ema_decay=0.999,
                 seed=5,
                 aug_type='',
                 decay_power=3,
                 alpha=1.,
                 aug_prob=1.):

        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.save_all_chkpts = save_all_chkpts
        self.writer = None
        self.use_ema_decay = use_ema_decay
        self.start_epoch = 0
        self.lr_finder = lr_finder
        self.fixbase_epoch = 0
        self.iter_to_wait = 0
        self.best_metric = 0.0
        self.max_epoch = None
        self.num_batches = None
        assert target_metric in ['train_loss', 'test_acc']
        self.target_metric = target_metric
        self.epoch = None
        self.train_patience = train_patience
        self.early_stopping = early_stopping
        self.state_cacher = StateCacher(in_memory=True, cache_dir=None)
        self.param_history = set()
        self.seed = seed
        self.models = OrderedDict()
        self.optims = OrderedDict()
        self.scheds = OrderedDict()
        self.ema_model = None
        if should_freeze_aux_models:
            print(f'Engine: should_freeze_aux_models={should_freeze_aux_models}')
        self.should_freeze_aux_models = should_freeze_aux_models
        self.nncf_metainfo = deepcopy(nncf_metainfo)
        self.compression_ctrl = compression_ctrl
        self.initial_lr = initial_lr
        self.epoch_interval_for_aux_model_freeze = epoch_interval_for_aux_model_freeze
        self.epoch_interval_for_turn_off_mutual_learning = epoch_interval_for_turn_off_mutual_learning
        self.model_names_to_freeze = []
        self.current_lr = None
        self.warmup_finished = True
        self.aug_type = aug_type
        self.alpha = alpha
        self.aug_prob = aug_prob
        self.aug_index = None
        self.lam = None
        self.decay_power = decay_power
        self.alpha = alpha

        if isinstance(models, (tuple, list)):
            assert isinstance(optimizers, (tuple, list))
            assert isinstance(schedulers, (tuple, list))

            num_models = len(models)
            assert len(optimizers) == num_models
            assert len(schedulers) == num_models

            for model_id, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
                model_name = 'main_model' if model_id == 0 else f'aux_model_{model_id}'
                self.register_model(model_name, model, optimizer, scheduler)
                if use_ema_decay and model_id == 0:
                    self.ema_model = ModelEmaV2(model, decay=ema_decay)
                if should_freeze_aux_models and model_id > 0:
                    self.model_names_to_freeze.append(model_name)
        else:
            assert not isinstance(optimizers, (tuple, list))
            assert not isinstance(schedulers, (tuple, list))
            assert not isinstance(models, (tuple, list))
            self.register_model('main_model', models, optimizers, schedulers)
            if use_ema_decay:
                self.ema_model = ModelEmaV2(models, decay=ema_decay)
        self.main_model_name = self.get_model_names()[0]
        self.scales = {}
        for model_name, model in self.models.items():
            scale = get_model_attr(model, 'scale')
            if not get_model_attr(model, 'use_angle_simple_linear') and  scale != 1.:
                print(f"WARNING:: Angle Linear is not used but the scale parameter in the loss {scale} != 1.")
            self.scales[model_name] = scale
        self.am_scale = self.scales[self.main_model_name] # for loss initialization
        assert initial_lr is not None
        self.lb_lr = initial_lr / lr_decay_factor
        self.per_batch_annealing = isinstance(self.scheds[self.main_model_name],
                                              (CosineAnnealingCycleRestart, OneCycleLR))

    def _should_freeze_aux_models(self, epoch):
        if not self.should_freeze_aux_models:
            return False
        if self.epoch_interval_for_aux_model_freeze is None:
            # simple case
            return True
        res = _get_cur_action_from_epoch_interval(self.epoch_interval_for_aux_model_freeze, epoch)
        print(f'_should_freeze_aux_models: return res={res}')
        return res

    def _should_turn_off_mutual_learning(self, epoch):
        if self.epoch_interval_for_turn_off_mutual_learning is None:
            # simple case
            return False
        res = _get_cur_action_from_epoch_interval(self.epoch_interval_for_turn_off_mutual_learning, epoch)
        print(f'_should_turn_off_mutual_learning: return {res}')
        return res

    def register_model(self, name='main_model', model=None, optim=None, sched=None):
        if self.__dict__.get('models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self.models[name] = model
        self.optims[name] = optim
        self.scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self.models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        return names_real

    def save_model(self, epoch, save_dir, is_best=False, should_save_ema_model=False):
        def create_sym_link(path,name):
            if osp.lexists(name):
                os.remove(name)
            os.symlink(path, name)

        names = self.get_model_names()
        for name in names:
            if should_save_ema_model and name == self.main_model_name:
                assert self.use_ema_decay
                model_state_dict = self.ema_model.module.state_dict()
            else:
                model_state_dict = self.models[name].state_dict()

            checkpoint = {
                'state_dict': model_state_dict,
                'epoch': epoch + 1,
                'optimizer': self.optims[name].state_dict(),
                'scheduler': self.scheds[name].state_dict(),
                'num_classes': self.datamanager.num_train_ids,
                'classes_map': self.datamanager.train_loader.dataset.classes,
                'initial_lr': self.initial_lr,
            }

            if self.compression_ctrl is not None and name == self.main_model_name:
                checkpoint['compression_state'] = self.compression_ctrl.get_compression_state()
                checkpoint['nncf_metainfo'] = self.nncf_metainfo

            ckpt_path = save_checkpoint(
                            checkpoint,
                            osp.join(save_dir, name),
                            is_best=is_best,
                            name=name
                        )

            if name == self.main_model_name:
                latest_ckpt_filename = 'latest.pth'
                best_ckpt_filename = 'best.pth'
            else:
                latest_ckpt_filename = f'latest_{name}.pth'
                best_ckpt_filename = f'best_{name}.pth'

            latest_name = osp.join(save_dir, latest_ckpt_filename)
            create_sym_link(ckpt_path, latest_name)
            if is_best:
                best_model = osp.join(save_dir, best_ckpt_filename)
                create_sym_link(ckpt_path, best_model)

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self.models[name].train()
            else:
                self.models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        lr = self.optims[name].param_groups[0]['lr']
        if isinstance(self.scheds[name], (WarmupScheduler, OneCycleLR)):
            return lr, self.scheds[name].warmup_finished
        return lr, True

    def update_lr(self, names=None, output_avg_metric=None):
        names = self.get_model_names(names)

        for name in names:
            if self.scheds[name] is not None:
                if isinstance(self.scheds[name], (ReduceLROnPlateauV2, WarmupScheduler)):
                    self.scheds[name].step(metrics=output_avg_metric)
                else:
                    self.scheds[name].step()

    def exit_on_plateau_and_choose_best(self, accuracy):
        '''
        The function returns a pair (should_exit, is_candidate_for_best).

        Default implementation of the method returns False for should_exit.
        Other behavior must be overridden in derived classes from the base Engine.
        '''

        is_candidate_for_best = False
        current_metric = np.round(accuracy, 4)
        if current_metric >= self.best_metric:
            self.best_metric = current_metric
            is_candidate_for_best = True

        return False, is_candidate_for_best

    def run(
        self,
        trial=None,
        save_dir='log',
        tb_writer=None,
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        topk=(1, 5, 10, 20),
        lr_finder=None,
        perf_monitor=None,
        stop_callback=None,
        initial_seed=5,
        **kwargs
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            topk (list, optional): cmc topk to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """

        if lr_finder:
            self.configure_lr_finder(trial, lr_finder)
            self.backup_model()

        self.save_dir = save_dir
        self.writer = tb_writer
        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        assert start_epoch != max_epoch, "the last epoch number cannot be equal the start one"
        if self.early_stopping or self.target_metric == 'test_acc':
            assert eval_freq == 1, "early stopping works only with evaluation on each epoch"
        self.fixbase_epoch = fixbase_epoch
        test_acc = AverageMeter()
        accuracy, should_save_ema_model = 0, False
        print('=> Start training')

        if perf_monitor and not lr_finder: perf_monitor.on_train_begin()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            # change the NumPy’s seed at every epoch
            np.random.seed(initial_seed + self.epoch)
            if perf_monitor and not lr_finder: perf_monitor.on_epoch_begin(self.epoch)
            if self.compression_ctrl is not None:
                self.compression_ctrl.scheduler.epoch_step(self.epoch)
            try:
                avg_loss = self.train(
                    print_freq=print_freq,
                    fixbase_epoch=fixbase_epoch,
                    open_layers=open_layers,
                    lr_finder=lr_finder,
                    perf_monitor=perf_monitor,
                    stop_callback=stop_callback
                )
            except RuntimeError as exp:
                print(f'Training has failed: {exp}')
                break

            if self.compression_ctrl is not None:
                statistics = self.compression_ctrl.statistics()
                print(statistics.to_str())
                if self.writer is not None and not lr_finder:
                    for key, value in get_nncf_prepare_for_tensorboard()(statistics).items():
                        self.writer.add_scalar(f"compression/statistics/{key}",
                                               value, len(self.train_loader) * self.epoch)

            if stop_callback and stop_callback.check_stop():
                break

            if (((self.epoch + 1) >= start_eval
               and eval_freq > 0
               and (self.epoch+1) % eval_freq == 0
               and (self.epoch + 1) != self.max_epoch)
               or self.epoch == (self.max_epoch - 1)):

                accuracy, should_save_ema_model = self.test(
                    self.epoch,
                    topk=topk,
                    lr_finder=lr_finder,
                )
            # update test_acc AverageMeter only if the accuracy is better than the average
            if accuracy >= test_acc.avg:
                test_acc.update(accuracy)

            target_metric = test_acc.avg if self.target_metric == 'test_acc' else avg_loss
            if perf_monitor and not lr_finder: perf_monitor.on_epoch_end(self.epoch, accuracy)

            if not lr_finder and not self.per_batch_annealing:
                self.update_lr(output_avg_metric = target_metric)

            if lr_finder:
                print(f"epoch: {self.epoch}\t accuracy: {accuracy}\t lr: {self.get_current_lr()[0]}")
                if trial:
                    trial.report(accuracy, self.epoch)
                    if trial.should_prune():
                        # restore model before pruning
                        self.restore_model()
                        raise optuna.exceptions.TrialPruned()

            if not lr_finder:
                # use smooth (average) accuracy metric for early stopping if the target metric is accuracy
                should_exit, is_candidate_for_best = self.exit_on_plateau_and_choose_best(accuracy)
                should_exit = self.early_stopping and should_exit

                if self.save_all_chkpts:
                    self.save_model(self.epoch, save_dir, is_best=is_candidate_for_best,
                                    should_save_ema_model=should_save_ema_model)
                elif is_candidate_for_best:
                    self.save_model(0, save_dir, is_best=is_candidate_for_best,
                                    should_save_ema_model=should_save_ema_model)

                if should_exit:
                    if self.compression_ctrl is None or \
                            (self.compression_ctrl is not None and
                                self.compression_ctrl.compression_stage() == \
                                    get_nncf_complession_stage().FULLY_COMPRESSED):
                        break

        if perf_monitor and not lr_finder: perf_monitor.on_train_end()
        if lr_finder and lr_finder.mode != 'fast_ai': self.restore_model()
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f'Elapsed {elapsed}')

        if self.writer is not None:
            self.writer.close()

        self._finalize_training()

        return accuracy, self.best_metric

    def _freeze_aux_models(self):
        for model_name in self.model_names_to_freeze:
            model = self.models[model_name]
            model.eval()
            open_specified_layers(model, [])

    def _unfreeze_aux_models(self):
        for model_name in self.model_names_to_freeze:
            model = self.models[model_name]
            model.train()
            open_all_layers(model)

    def configure_lr_finder(self, trial, finder_cfg):
        if trial is None:
            return
        lr = trial.suggest_float("lr", finder_cfg.min_lr, finder_cfg.max_lr, step=finder_cfg.step)
        if lr in self.param_history:
            # restore model before pruning
            self.restore_model()
            raise optuna.exceptions.TrialPruned()
        self.param_history.add(lr)
        for param_group in self.optims[self.main_model_name].param_groups:
            param_group["lr"] = round(lr,6)
        print(f"training with next lr: {lr}")

    def backup_model(self):
        print("backuping model...")
        model_device = next(self.models[self.main_model_name].parameters()).device
        # explicitly put the model on the CPU before storing it in memory
        self.state_cacher.store(key="model",
                                state_dict=get_model_attr(self.models[self.main_model_name], 'cpu')().state_dict())
        self.state_cacher.store(key="optimizer", state_dict=self.optims[self.main_model_name].state_dict())
        # restore the model device
        get_model_attr(self.models[self.main_model_name],'to')(model_device)

    def restore_model(self):
        print("restoring model and seeds to initial state...")
        model_device = next(self.models[self.main_model_name].parameters()).device
        get_model_attr(self.models[self.main_model_name], 'load_state_dict')(self.state_cacher.retrieve("model"))
        self.optims[self.main_model_name].load_state_dict(self.state_cacher.retrieve("optimizer"))
        get_model_attr(self.models[self.main_model_name],'to')(model_device)
        set_random_seed(self.seed)

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None, lr_finder=False, perf_monitor=None,
              stop_callback=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        accuracy = AverageMeter()

        self.set_model_mode('train')

        if not self._should_freeze_aux_models(self.epoch):
            # NB: it should be done before `two_stepped_transfer_learning`
            # to give possibility to freeze some layers in the unlikely event
            # that `two_stepped_transfer_learning` is used together with nncf
            self._unfreeze_aux_models()

        #self.two_stepped_transfer_learning(
        #    self.epoch, fixbase_epoch, open_layers
        #)

        if self._should_freeze_aux_models(self.epoch):
            self._freeze_aux_models()

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            if perf_monitor and not lr_finder: perf_monitor.on_train_batch_begin(self.batch_idx)

            data_time.update(time.time() - end)

            if self.compression_ctrl:
                self.compression_ctrl.scheduler.step(self.batch_idx)

            loss_summary, avg_acc = self.forward_backward(data)
            batch_time.update(time.time() - end)
            last_main_loss = loss_summary[self.get_model_names()[0]]
            if math.isnan(last_main_loss) or math.isinf(last_main_loss):
                raise RuntimeError('Loss is NaN or Inf, exiting the training...')

            losses.update(loss_summary)
            accuracy.update(avg_acc)
            if perf_monitor and not lr_finder: perf_monitor.on_train_batch_end(self.batch_idx)

            if not lr_finder and (((self.batch_idx + 1) % print_freq) == 0 or
                                        self.batch_idx == self.num_batches - 1):
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (self.max_epoch - (self.epoch + 1)) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f'epoch: [{self.epoch + 1}/{self.max_epoch}][{self.batch_idx + 1}/{self.num_batches}]\t'
                    f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'cls acc {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                    f'eta {eta_str}\t'
                    f'{losses}\t'
                    f'lr {self.get_current_lr()[0]:.6f}'
                )

            if self.writer is not None and not lr_finder:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                self.writer.add_scalar('Aux/lr', self.get_current_lr()[0], n_iter)
                self.writer.add_scalar('Accuracy/train', accuracy.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Loss/' + name, meter.avg, n_iter)

            end = time.time()
            self.current_lr, self.warmup_finished = self.get_current_lr()
            if stop_callback and stop_callback.check_stop():
                break
            if not lr_finder and self.use_ema_decay:
                self.ema_model.update(self.models[self.main_model_name])
            if self.per_batch_annealing:
                self.update_lr()

        return losses.meters['loss'].avg

    @abc.abstractmethod
    def forward_backward(self, data):
        pass

    def _apply_batch_augmentation(self, imgs):
        def rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2

        if self.aug_type == 'fmix':
            r = np.random.rand(1)
            if self.alpha > 0 and r[0] <= self.aug_prob:
                lam, fmask = sample_mask(self.alpha, self.decay_power, imgs.shape[-2:])
                index = torch.randperm(imgs.size(0), device=imgs.device)
                fmask = torch.from_numpy(fmask).float().to(imgs.device)
                # Mix the images
                x1 = fmask * imgs
                x2 = (1 - fmask) * imgs[index]
                self.aug_index = index
                self.lam = lam
                imgs = x1 + x2
            else:
                self.aug_index = None
                self.lam = None

        elif self.aug_type == 'mixup':
            r = np.random.rand(1)
            if self.alpha > 0 and r <= self.aug_prob:
                lam = np.random.beta(self.alpha, self.alpha)
                index = torch.randperm(imgs.size(0), device=imgs.device)

                imgs = lam * imgs + (1 - lam) * imgs[index, :]
                self.lam = lam
                self.aug_index = index
            else:
                self.aug_index = None
                self.lam = None

        elif self.aug_type == 'cutmix':
            r = np.random.rand(1)
            if self.alpha > 0 and r <= self.aug_prob:
                # generate mixed sample
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(imgs.size(0), device=imgs.device)

                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                self.lam = lam
                self.aug_index = rand_index
            else:
                self.aug_index = None
                self.lam = None

        return imgs

    def test(
        self,
        epoch,
        topk=(1, 5, 10, 20),
        lr_finder = False,
        test_only=False,
        **kwargs
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """

        self.set_model_mode('eval')
        models_to_eval = list(self.models.items())
        top1=[]
        if (self.use_ema_decay and not lr_finder and not test_only):
            models_to_eval.append(('EMA model', self.ema_model.module))

        print('##### Evaluating test dataset #####')
        for model_name, model in models_to_eval:
            # do not evaluate second model till last epoch
            if (model_name not in [self.main_model_name, 'EMA model']
                    and not test_only and epoch != (self.max_epoch - 1)):
                continue
            # we may compute some other metric here, but consider it as top1 for consistency
            # with single label classification
            cur_top1 = self._evaluate(
                model=model,
                epoch=epoch,
                data_loader=self.test_loader,
                model_name=model_name,
                topk=topk,
                lr_finder=lr_finder,
                **kwargs
            )
            if model_name in [self.main_model_name, 'EMA model']:
                top1.append(cur_top1)
        max_top1 = max(top1)
        return max_top1, top1.index(max_top1)

    @staticmethod
    def parse_data_for_train(data, use_gpu=False):
        imgs = data[0]
        obj_ids = data[1]
        if use_gpu:
            imgs = imgs.cuda()
            obj_ids = obj_ids.cuda()
        return imgs, obj_ids

    @staticmethod
    def parse_data_for_eval(data):
        imgs = data[0]
        obj_ids = data[1]
        cam_ids = data[2]
        return imgs, obj_ids, cam_ids

    def two_stepped_transfer_learning(self, epoch, fixbase_epoch, open_layers):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(f'* Only train {open_layers} (epoch: {epoch + 1}/{fixbase_epoch})')

            for model in self.models.values():
                open_specified_layers(model, open_layers, strict=False)
        else:
            for model in self.models.values():
                open_all_layers(model)

    @abc.abstractmethod
    def _evaluate(self, model, epoch, data_loader, model_name, topk, lr_finder):
        return 0.

    @abc.abstractmethod
    def _finalize_training(self):
        pass
