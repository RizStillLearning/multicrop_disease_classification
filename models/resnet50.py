import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck as TorchvisionBottleneck
from .cbam import CBAM

class CBAttentionBottleneck(nn.Module):
    """Bottleneck with CBAM attention that matches torchvision's ResNet API"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, use_cbam=True):
        super(CBAttentionBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(channel_in=planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def get_cbam_resnet50(num_classes=10, pretrained=True):
    """Create a ResNet50 with CBAM attention modules"""
    model = ResNet(block=CBAttentionBottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)

    if pretrained:
        state_dict = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).state_dict()
        # Remove fc layer weights as they have different dimensions (1000 vs num_classes)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)

    return model