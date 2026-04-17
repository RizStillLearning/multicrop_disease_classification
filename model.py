import os
import torch
import torch.nn as nn
from utils import get_device
from torchvision import models

class CropDiseaseClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=16):
        super(CropDiseaseClassifier, self).__init__()
        models_dict = {
            'resnet50': models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) if pretrained else models.resnet50(weights=None),
            'vgg19': models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1) if pretrained else models.vgg19(weights=None),
            'convnext_large': models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1) if pretrained else models.convnext_large(weights=None),
            'efficientnet_b0': models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) if pretrained else models.efficientnet_b0(weights=None),
        }

        if model_name not in models_dict:
            raise ValueError(f"Unsupported model name: {model_name}. Supported models are: {list(models_dict.keys())}")
        
        self.base_model = models_dict[model_name]
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        try:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        except AttributeError:
            in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier[-1] = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.head(x)
        return x
    
def build_model(model_name, num_classes=16):
    model = CropDiseaseClassifier(model_name=model_name, num_classes=num_classes)
    return model

def save_model(model_name, model, file_name='model.pth', **config):
    os.makedirs(model_name, exist_ok=True)
    config['model_state_dict'] = model.state_dict()
    save_path = os.path.join(model_name, file_name)
    torch.save(config, save_path)

def load_model(model_name, model, file_name='model.pth'):
    path = os.path.join(model_name, file_name)
    config = torch.load(path, map_location=get_device(), weights_only=True)
    model.load_state_dict(config['model_state_dict'])
    crop_disease_classes = config['crop_disease_classes']
    return crop_disease_classes