import torch
import torch.nn as nn
from torchvision import models
from .cbam import CBAM
from torchvision.models.vgg import VGG, make_layers

class VGG19CBAM(nn.Module):
    """VGG19 with CBAM attention modules inserted after each convolutional block."""

    def __init__(self, num_classes=16, pretrained=True, use_cbam=True):
        super(VGG19CBAM, self).__init__()
        self.use_cbam = use_cbam

        if pretrained:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            vgg = models.vgg19(weights=None)

        self.features = self._make_cbam_features(vgg.features) if self.use_cbam else vgg.features
        self.avgpool = vgg.avgpool

        classifier = list(vgg.classifier.children())
        classifier[-1] = nn.Linear(in_features=classifier[-1].in_features, out_features=num_classes)
        self.classifier = nn.Sequential(*classifier)

    def _make_cbam_features(self, features):
        block_ends = {4: 64, 9: 128, 18: 256, 27: 512, 36: 512}
        layers = []

        for idx, module in enumerate(features):
            layers.append(module)
            if idx in block_ends:
                layers.append(CBAM(channel_in=block_ends[idx]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_cbam_vgg19(num_classes=16, pretrained=True, use_cbam=True):
    """Create a VGG19 model with CBAM attention modules."""
    return VGG19CBAM(num_classes=num_classes, pretrained=pretrained, use_cbam=use_cbam)
