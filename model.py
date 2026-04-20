import os
import torch
import torch.nn as nn
from utils import get_device
from models.efficientnet_b0 import get_cbam_efficientnet_b0

def build_model(num_classes=16):
    return get_cbam_efficientnet_b0(num_classes=num_classes)

def save_model(final_model_dir, final_model_name, model, **config):
    os.makedirs(final_model_dir, exist_ok=True)
    config['model_state_dict'] = model.state_dict()
    save_path = os.path.join(final_model_dir, final_model_name)
    torch.save(config, save_path)

def load_model(final_model_path, model):
    config = torch.load(final_model_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(config['model_state_dict'])
    crop_disease_classes = config['crop_disease_classes']
    return crop_disease_classes