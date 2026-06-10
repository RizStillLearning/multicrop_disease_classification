import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121
from .cbam import CBAM


class DenseBlockWithCBAM(nn.Module):
    '''Wraps a DenseNet DenseBlock with a CBAM module applied after it.'''

    def __init__(self, dense_block, num_output_features):
        super(DenseBlockWithCBAM, self).__init__()
        self.dense_block = dense_block
        self.cbam = CBAM(channel_in=num_output_features)

    def forward(self, x):
        out = self.dense_block(x)
        out = self.cbam(out)
        return out


def _get_densenet121_out_channels(block, in_channels, growth_rate, num_layers):
    '''Compute the output channel count of a DenseBlock.'''
    return in_channels + num_layers * growth_rate


def insert_cbam_into_densenet(model):
    '''
    Insert CBAM after every DenseBlock in DenseNet-121.
    DenseNet-121 layer structure inside model.features:
      conv0, norm0, relu0, pool0,
      denseblock1, transition1,
      denseblock2, transition2,
      denseblock3, transition3,
      denseblock4,
      norm5
    Each DenseBlock output channel count is tracked cumulatively.
    '''
    growth_rate = model.growth_rate          # 32 for DenseNet-121
    # DenseNet-121 block config: [6, 12, 24, 16]
    block_config = model.block_config        # tuple of num_layers per block

    # Initial number of features after stem (conv0 + pool0)
    in_channels = model.features.conv0.out_channels  # 64

    feature_names = list(model.features._modules.keys())

    for name in feature_names:
        if name.startswith('denseblock'):
            block_idx = int(name[-1]) - 1   # 0-indexed
            num_layers = block_config[block_idx]
            out_channels = in_channels + num_layers * growth_rate

            original_block = model.features._modules[name]
            model.features._modules[name] = DenseBlockWithCBAM(
                dense_block=original_block,
                num_output_features=out_channels
            )
            in_channels = out_channels

        elif name.startswith('transition'):
            # Transition halves the channels
            transition = model.features._modules[name]
            # The transition's conv determines the new channel count
            in_channels = transition.conv.out_channels

    return model


def get_densenet121(num_classes=16, pretrained=True):
    '''Plain DenseNet-121 without CBAM.'''
    model = densenet121(
        weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.classifier.in_features   # 1024
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def get_cbam_densenet121(num_classes=16, pretrained=True):
    '''DenseNet-121 with CBAM inserted after every DenseBlock.'''
    model = densenet121(
        weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    )
    insert_cbam_into_densenet(model)

    in_features = model.classifier.in_features   # 1024
    model.classifier = nn.Linear(in_features, num_classes)
    return model
