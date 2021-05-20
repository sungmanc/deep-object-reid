
import math
import os
from os import path as osp
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2 as cv
import torch
import torch.utils as utils
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sc_sdk.entities.datasets import Dataset as NousDataset, Subset, DatasetItem
from sc_sdk.entities.label import Label, ScoredLabel
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.reporting.callback import Callback

from torchreid.models.common import ModelInterface


def generate_batch_indices(count, batch_size):
    for i in range(math.ceil(count / batch_size)):
        yield slice(i * batch_size, (i + 1) * batch_size)


class CannotLoadModelException(ValueError):
    pass


logger = logger_factory.get_logger("TorchClassificationInstance")


class ClassificationImageFolder():
    def __init__(self, nous_dataset, labels):
        super().__init__()
        self.nous_dataset = nous_dataset
        self.labels = labels
        self.annotation = []

        for i in range(len(self.nous_dataset)):
            if self.nous_dataset[i].annotation.get_labels():
                label = self.nous_dataset[i].annotation.get_labels()[0]
                class_num = self.labels.index(label)
            else:
                class_num = 0
            self.annotation.append({'label': class_num})

    def __getitem__(self, idx):
        sample = self.nous_dataset[idx].numpy  # This returns 8-bit numpy array of shape (height, width, RGB)
        label = self.annotation[idx]['label']
        return {'img': sample, 'label': label}

    def __len__(self):
        return len(self.annotation)

    def get_annotation(self):
        return self.annotation

    def get_classes(self):
        return self.labels


class ClassificationDataloader(Dataset):
    """
    Dataloader that generates logits from DatasetItems.
    """

    def __init__(self, dataset: Union[NousDataset, List[DatasetItem]],
                 labels: List[Label],
                 inference_mode: bool = False,
                 augmentation: Callable = None):
        self.dataset = dataset
        self.labels = labels
        self.inference_mode = inference_mode
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset)

    def get_input(self, idx: int):
        """
        Return the centered and scaled input tensor for file with 'idx'
        """
        sample = self.dataset[idx].numpy  # This returns 8-bit numpy array of shape (height, width, RGB)

        if self.augmentation is not None:
            img = Image.fromarray(sample)
            img, _ = self.augmentation((img, ''))
        return img

    def __getitem__(self, idx: int):
        """
        Return the input and the an optional encoded target for training with index 'idx'
        """
        input_image = self.get_input(idx)
        _, h, w = input_image.shape

        if self.inference_mode:
            class_num = np.asarray(0)
        else:
            item = self.dataset[idx]
            if len(item.annotation.get_labels()) == 0:
                raise ValueError(
                    f"No labels in annotation found. Annotation: {item.annotation}")
            label = item.annotation.get_labels()[0]
            class_num = self.labels.index(label)
            class_num = np.asarray(class_num)
        return input_image, class_num.astype(np.float32)

@torch.no_grad()
def predict(dataset_slice: List[DatasetItem], labels: List[Label], model: ModelInterface,
            transform, device: torch.device) -> List[Tuple[DatasetItem, List[ScoredLabel], np.array]]:
    """
    Predict from a list of 'DatasetItem' using 'model'. Scale image prior to inference to 'image_shape'
    :return: Return a list of tuple instances, that hold the resulting DatasetItem, ScoredLabels
    and the saliency map generated by Gradcam++
    """
    model.eval()
    model.to(device)
    instances_per_image = list()
    logger.info("Predicting {} files".format(len(dataset_slice)))

    d_set = ClassificationDataloader(dataset=dataset_slice, labels=labels, augmentation=transform, inference_mode=True)
    loader = utils.data.DataLoader(d_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for inputs, y in loader:  # tqdm
        inputs = torch.tensor(inputs, device=device)
        outputs = model(inputs)[0]
        outputs = outputs.cpu().detach().numpy()

        # Multiclass
        for i, output in enumerate(outputs):
            dataset_item = dataset_slice[i]
            class_num = int(np.argmax(output))
            class_prob = float(outputs[i, class_num].squeeze())
            label = ScoredLabel(label=labels[class_num], probability=class_prob)
            scored_labels = [label]
            dataset_item.append_labels(labels=scored_labels)
            instances_per_image.append((dataset_item, scored_labels))

    return instances_per_image

def list_available_models(models_directory):
    available_models = []
    for dirpath, dirnames, filenames in os.walk(models_directory):
        for filename in filenames:
            if filename == 'main_model.yaml':
                available_models.append(dict(
                    name=osp.basename(dirpath),
                    dir=osp.join(models_directory, dirpath)))
    available_models.sort(key=lambda x: x['name'])
    return available_models
