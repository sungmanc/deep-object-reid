# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import logging
import os.path as osp
import sys
from datetime import datetime
import time

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment

from torchreid.integration.sc.utils import (ClassificationDatasetAdapter,
                                            generate_label_schema,
                                            get_task_class)
from torchreid.utils.avgmeter import AverageMeter


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument('--steps', nargs='+')
    parser.add_argument('--debug-dump-folder', default='')
    args = parser.parse_args()
    return args


def main(args):
    if args.debug_dump_folder:
        from torchreid.utils import Logger
        log_name = 'ote_task.log' + time.strftime('-%Y-%m-%d-%H-%M-%S')
        sys.stdout = Logger(osp.join(args.debug_dump_folder, log_name))
    weights = None
    logger.info('Initialize dataset')
    metrics = dict()
    times = dict()
    for step in args.steps:
        data_root_ = osp.join(args.data_dir, step)
        dataset = ClassificationDatasetAdapter(
            train_data_root=osp.join(data_root_, 'train'),
            train_ann_file=osp.join(data_root_, 'train.json'),
            val_data_root=osp.join(data_root_, 'val'),
            val_ann_file=osp.join(data_root_, 'val.json'),
            test_data_root=osp.join(data_root_, 'val'),
            test_ann_file=osp.join(data_root_, 'val.json')
            )
        logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
        logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')
        logger.info('Train model')

        labels_schema = generate_label_schema(dataset.get_labels(), dataset.is_multilabel())
        logger.info('Load model template')
        model_template = parse_model_template(args.template_file_path)

        logger.info('Set hyperparameters')

        params = create(model_template.hyper_parameters.data)
        logger.info('Setup environment')
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema, model_template=model_template)
        logger.info('Create base Task')

        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)

        start_time = datetime.now()
        output_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        if args.raw:
            task.train(dataset, output_model, weights=None)
        else:
            weights = task.train(dataset, output_model, weights=weights)
        training_time = datetime.now() - start_time
        print("Training time: ", training_time)
        logger.info('Get predictions on the validation set')
        validation_dataset = dataset.get_subset(Subset.VALIDATION)
        predicted_validation_dataset = task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Estimate quality on validation set')
        task.evaluate(resultset)
        print(resultset.performance)
        logger.info(str(resultset.performance))
        times[step] = training_time
        metrics[step] = str(resultset.performance)

        if args.export:
            logger.info('Export model')
            exported_model = ModelEntity(
                dataset,
                environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)
            task.export(ExportType.OPENVINO, exported_model)

            logger.info('Create OpenVINO Task')
            environment.model = exported_model
            openvino_task_impl_path = model_template.entrypoints.openvino
            openvino_task_cls = get_task_class(openvino_task_impl_path)
            openvino_task = openvino_task_cls(environment)

            logger.info('Get predictions on the validation set')
            predicted_validation_dataset = openvino_task.infer(
                validation_dataset.with_empty_annotations(),
                InferenceParameters(is_evaluation=True))
            resultset = ResultSetEntity(
                model=output_model,
                ground_truth_dataset=validation_dataset,
                prediction_dataset=predicted_validation_dataset,
            )
            logger.info('Estimate quality on validation set')
            openvino_task.evaluate(resultset)
            logger.info(str(resultset.performance))

            logger.info('Run POT optimization')
            optimized_model = ModelEntity(
                dataset,
                environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)
            openvino_task.optimize(
                OptimizationType.POT,
                dataset.get_subset(Subset.TRAINING),
                optimized_model,
                OptimizationParameters())

            logger.info('Get predictions on the validation set')
            predicted_validation_dataset = openvino_task.infer(
                validation_dataset.with_empty_annotations(),
                InferenceParameters(is_evaluation=True))
            resultset = ResultSetEntity(
                model=optimized_model,
                ground_truth_dataset=validation_dataset,
                prediction_dataset=predicted_validation_dataset,
            )
            logger.info('Performance of optimized model:')
            openvino_task.evaluate(resultset)
            logger.info(str(resultset.performance))
    print("TIME: ", times)
    print("METRIC: ", metrics)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    sys.exit(main(args) or 0)
