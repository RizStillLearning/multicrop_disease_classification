import os
import torch
import torch.nn as nn
from utils import get_device
from models.resnet50 import get_cbam_resnet50
from models.vgg19 import get_cbam_vgg19

def build_model(model_name, num_classes=16):
    if model_name == 'resnet50':
        return get_cbam_resnet50(num_classes=num_classes)
    elif model_name == 'vgg19':
        return get_cbam_vgg19(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are: ['resnet50']")

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