import os
import torch
import torch.nn as nn
from utils import get_device
from models.efficientnet_b0 import get_cbam_efficientnet_b0

def build_model(num_classes=16):
    return get_cbam_efficientnet_b0(num_classes=num_classes)

def save_model(model_dir, model_name, model, **config):
    os.makedirs(model_dir, exist_ok=True)
    config['model_state_dict'] = model.state_dict()
    save_path = os.path.join(model_dir, model_name)
    torch.save(config, save_path)

def load_model(model_path, model):
    config = torch.load(model_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(config['model_state_dict'])
    crop_disease_classes = config['crop_disease_classes']
    return crop_disease_classes

def extract_features(model, dataloader, device):
    batch_features = []
    batch_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            feature = model(images)
            batch_features.append(feature.cpu())
            batch_labels.append(labels.cpu())

    features = torch.cat(batch_features, dim=0)
    labels = torch.cat(batch_labels, dim=0)
    return features, labels
