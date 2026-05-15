import os
import torch
from core.utils import get_device
from models.efficientnet_b0 import get_cbam_efficientnet_b0

def build_model(num_classes=16):
    return get_cbam_efficientnet_b0(num_classes=num_classes)

def save_model(model_dir, model_name, model):
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, model_name)
    save_config = dict(model_state_dict=model.state_dict())
    torch.save(save_config, save_path)

def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])