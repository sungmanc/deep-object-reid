# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import

from .efficient_net_pytcv import *
from .inceptionv4_pytcv import *
from .mobilenetv3 import *
from .ptcv_wrapper import *
from .timm_wrapper import *
from .q2l import *
from .transformer import *
from .gcn import *
from .ml_decoder import *
from .resnet import *

__model_factory = {
    # image classification models
    'inceptionv4': inceptionv4_pytcv,
    'mobilenetv3_small': mobilenetv3_small,
    'mobilenetv3_large': mobilenetv3_large,
    'mobilenetv3_large_075': mobilenetv3_large_075,
    'mobilenetv3_large_150': mobilenetv3_large_150,
    'mobilenetv3_large_125': mobilenetv3_large_125,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2b,
    'efficientnet_b3': efficientnet_b3b,
    'efficientnet_b4': efficientnet_b4b,
    'efficientnet_b5': efficientnet_b5b,
    'efficientnet_b6': efficientnet_b6b,
    'efficientnet_b7': efficientnet_b7b,
    'resnet50': resnet50,
}

__model_factory = {**__model_factory, **pytcv_wrapped_models, **timm_wrapped_models}


def build_model(name, **kwargs):
    """A function wrapper for building a model.

    Args:
        name (str): model name.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name.startswith('q2l'):
        backbone_name = name[4:]
        if backbone_name not in avai_models:
            raise KeyError('Unknown backbone for Q2L model: {}. Must be one of {}'.format(backbone_name, avai_models))
        backbone = __model_factory[backbone_name](**kwargs)
        transformer = build_transformer(hidden_dim=backbone.num_features, dim_feedforward=backbone.num_features, **kwargs)
        model = build_q2l(backbone, transformer, hidden_dim=backbone.num_features, **kwargs)
    elif name.startswith('gcn'):
        backbone_name = name[4:]
        if backbone_name not in avai_models:
            raise KeyError('Unknown backbone for GCN model: {}. Must be one of {}'.format(backbone_name, avai_models))
        backbone = __model_factory[backbone_name](**kwargs)
        model = build_image_gcn(backbone, **kwargs)
    elif name.startswith('mld'):
        backbone_name = name[4:]
        if backbone_name not in avai_models:
            raise KeyError('Unknown backbone for MLD model: {}. Must be one of {}'.format(backbone_name, avai_models))
        backbone = __model_factory[backbone_name](**kwargs)
        model = build_ml_decoder_model(backbone, **kwargs)
    elif name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    else:
        model = __model_factory[name](**kwargs)
    return model
