import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.ops.misc import SqueezeExcitation
from .cbam import CBAM

# Example of a dynamic swap function
def replace_se_with_cbam(model):
    for name, module in model.named_children():
        if isinstance(module, SqueezeExcitation):
            # Fetch the input channels to initialize CBAM correctly
            in_channels = module.fc1.in_channels
            # Replace the SE layer with CBAM
            setattr(model, name, CBAM(channel_in=in_channels))
        else:
            # Recursively apply to child modules
            replace_se_with_cbam(module)

def get_cbam_efficientnet_b0(num_classes=16, pretrained=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    replace_se_with_cbam(model)

    # Replace the final classifier layer to match the number of classes
    in_features = model.classifier[1].in_features  # Get the input features of the last layer
    model.classifier[1] = nn.Linear(in_features, num_classes)  # Replace with new linear layer

    return model