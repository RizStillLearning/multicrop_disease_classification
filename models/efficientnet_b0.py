import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.ops.misc import SqueezeExcitation
from .cbam import CBAM, CBAM_V2

# Example of a dynamic swap function
def replace_se_with_cbam(model, use_v2=False):
    cbam_class = CBAM_V2 if use_v2 else CBAM
    for name, module in model.named_children():
        if isinstance(module, SqueezeExcitation):
            # Fetch the input channels to initialize CBAM correctly
            in_channels = module.fc1.in_channels
            # Replace the SE layer with CBAM
            setattr(model, name, cbam_class(channel_in=in_channels))
        else:
            # Recursively apply to child modules
            replace_se_with_cbam(module, use_v2=use_v2)

def get_cbam_efficientnet_b0(num_classes=16, pretrained=True, use_v2=False):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    replace_se_with_cbam(model, use_v2=use_v2)

    # Replace the final classifier layer to match the number of classes
    in_features = model.classifier[1].in_features  # Get the input features of the last layer
    model.classifier[1] = nn.Linear(in_features, num_classes)  # Replace with new linear layer

    return model

def get_cbam_v2_efficientnet_b0(num_classes=16, pretrained=True):
    '''Get EfficientNet-B0 with improved CBAM (V2) modules replacing SE blocks.
    '''
    return get_cbam_efficientnet_b0(num_classes=num_classes, pretrained=pretrained, use_v2=True)