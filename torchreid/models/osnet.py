from __future__ import division, absolute_import

import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout


__all__ = [
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0'
]

pretrained_urls = {
    'osnet_x1_0':
    'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'osnet_x0_75':
    'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'osnet_x0_5':
    'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'osnet_x0_25':
    'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_ibn_x1_0':
    'https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l'
}


##########
# Basic layers
##########

class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########

class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


##########
# Network architecture
##########

class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        IN=False,
        feature_dim=512,
        loss='softmax',
        attr_tasks=None,
        enable_attr_tasks=False,
        num_parts=None,
        **kwargs
    ):
        super(OSNet, self).__init__()

        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1

        self.loss = loss
        self.feature_dim = feature_dim
        assert self.feature_dim is not None and self.feature_dim > 0

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )

        out_num_channels = channels[3]
        self.conv5 = Conv1x1(channels[3], out_num_channels)

        if isinstance(num_classes, (list, tuple)):
            assert len(num_classes) == 2
            real_data_num_classes, synthetic_data_num_classes = num_classes
        else:
            real_data_num_classes, synthetic_data_num_classes = num_classes, None

        classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear
        self.num_parts = num_parts if num_parts is not None and num_parts > 1 else 0

        if self.num_parts > 1:
            self.part_self_fc = nn.ModuleList()
            self.part_rest_fc = nn.ModuleList()
            self.part_cat_fc = nn.ModuleList()
            for _ in range(self.num_parts):
                self.part_self_fc.append(self._construct_fc_layer(out_num_channels, out_num_channels))
                self.part_rest_fc.append(self._construct_fc_layer(out_num_channels, out_num_channels))
                self.part_cat_fc.append(self._construct_fc_layer(2 * out_num_channels, out_num_channels))

        fc_layers, classifier_layers = [], []
        for _ in range(self.num_parts + 1):  # main branch + part-based branches
            fc_layers.append(self._construct_fc_layer(out_num_channels, self.feature_dim, dropout=False))
            classifier_layers.append(classifier_block(self.feature_dim, real_data_num_classes))
        self.fc = nn.ModuleList(fc_layers)
        self.classifier = nn.ModuleList(classifier_layers)

        self.aux_fc = None
        self.aux_classifier = None
        self.split_embeddings = synthetic_data_num_classes is not None
        if self.split_embeddings:
            aux_fc_layers, aux_classifier_layers = [], []
            for _ in range(self.num_parts + 1):  # main branch + part-based branches
                aux_fc_layers.append(self._construct_fc_layer(out_num_channels, self.feature_dim, dropout=False))
                aux_classifier_layers.append(classifier_block(self.feature_dim, synthetic_data_num_classes))
            self.aux_fc = nn.ModuleList(aux_fc_layers)
            self.aux_classifier = nn.ModuleList(aux_classifier_layers)

        self.attr_fc = None
        self.attr_classifiers = None
        if enable_attr_tasks and attr_tasks is not None and len(attr_tasks) > 0:
            attr_fc = dict()
            attr_classifier = dict()
            for attr_name, attr_num_classes in attr_tasks.items():
                attr_fc[attr_name] = self._construct_fc_layer(out_num_channels, self.feature_dim // 4, dropout=False)
                attr_classifier[attr_name] = AngleSimpleLinear(self.feature_dim // 4, attr_num_classes)
            self.attr_fc = nn.ModuleDict(attr_fc)
            self.attr_classifiers = nn.ModuleDict(attr_classifier)

        self._init_params()

    @staticmethod
    def _make_layer(block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = [block(in_channels, out_channels, IN=IN)]
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    @staticmethod
    def _construct_fc_layer(input_dim, out_dim, dropout=False):
        layers = []

        if dropout:
            layers.append(Dropout(p=0.5, dist='gaussian'))

        layers.extend([
            nn.Linear(input_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ])

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _backbone(self, x):
        y = self.conv1(x)
        y = self.maxpool(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)

        return y

    def _glob_feature_vector(self, x, num_parts):
        return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

    def _part_feature_vector(self, x, num_parts):
        if num_parts <= 1:
            return []

        gap_branch = F.adaptive_avg_pool2d(x, (num_parts, 1)).squeeze(dim=-1)
        gmp_branch = F.adaptive_max_pool2d(x, (num_parts, 1)).squeeze(dim=-1)
        feature_vectors = gap_branch + gmp_branch

        return [f.squeeze(dim=-1) for f in torch.split(feature_vectors, 1, dim=-1)]

    def forward(self, x, return_featuremaps=False, get_embeddings=False, return_logits=False):
        feature_maps = self._backbone(x)
        if return_featuremaps:
            return feature_maps

        glob_feature = self._glob_feature_vector(feature_maps, num_parts=self.num_parts)
        part_features = self._part_feature_vector(feature_maps, num_parts=self.num_parts)
        features = [glob_feature] + list(part_features)

        main_embeddings = [fc(f) for f, fc in zip(features, self.fc)]
        if not self.training and not return_logits:
            return torch.cat(main_embeddings, dim=-1)

        main_logits = [classifier(embd) for embd, classifier in zip(main_embeddings, self.classifier)]
        main_centers = [classifier.get_centers() for classifier in self.classifier]

        if self.split_embeddings:
            aux_embeddings = [fc(f) for f, fc in zip(features, self.aux_fc)]
            aux_logits = [classifier(embd) for embd, classifier in zip(aux_embeddings, self.aux_classifier)]
            aux_centers = [classifier.get_centers() for classifier in self.aux_classifier]
        else:
            aux_embeddings = [None] * len(features)
            aux_logits = [None] * len(features)
            aux_centers = [None] * len(features)

        all_embeddings = dict(real=main_embeddings, synthetic=aux_embeddings)
        all_outputs = dict(real=main_logits, synthetic=aux_logits,
                           real_centers=main_centers, synthetic_centers=aux_centers)

        attr_embeddings = dict()
        if self.attr_fc is not None:
            for attr_name, attr_fc in self.attr_fc.items():
                attr_embeddings[attr_name] = attr_fc(glob_feature)

        attr_logits = dict()
        if self.attr_classifiers is not None:
            for att_name, attr_classifier in self.attr_classifiers.items():
                attr_logits[att_name] = attr_classifier(attr_embeddings[att_name])

        if get_embeddings:
            return all_embeddings, all_outputs, attr_logits

        if self.loss in ['softmax', 'am_softmax']:
            return all_outputs, attr_logits
        elif self.loss in ['triplet']:
            return all_outputs, attr_logits, all_embeddings
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

    def load_pretrained_weights(self, pretrained_dict):
        model_dict = self.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in pretrained_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            warnings.warn(
                'The pretrained weights cannot be loaded, '
                'please check the key names manually '
                '(** ignored and continue **)'
            )
        else:
            print('Successfully loaded pretrained weights')
            if len(discarded_layers) > 0:
                print(
                    '** The following layers are discarded '
                    'due to unmatched keys or layer size: {}'.
                        format(discarded_layers)
                )


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model.load_pretrained_weights(state_dict)


##########
# Instantiation
##########

def osnet_x1_0(num_classes, pretrained=False, download_weights=False, **kwargs):
    # standard size (width x1.0)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_x1_0')

    return model


def osnet_x0_75(num_classes, pretrained=False, download_weights=False, **kwargs):
    # medium size (width x0.75)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_x0_75')

    return model


def osnet_x0_5(num_classes, pretrained=False, download_weights=False, **kwargs):
    # tiny size (width x0.5)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_x0_5')

    return model


def osnet_x0_25(num_classes, pretrained=False, download_weights=False, **kwargs):
    # very tiny size (width x0.25)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_x0_25')

    return model


def osnet_ibn_x1_0(num_classes, pretrained=False, download_weights=False, **kwargs):
    # standard size (width x1.0) + IBN layer
    # Ref: Pan et al. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV, 2018.
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        IN=True,
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_ibn_x1_0')

    return model
