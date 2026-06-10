"""
predict.py — DenseNet-121 (backbone ensemble) inference

Usage (from project root):
    python densenet121/scripts/predict.py <path_to_image>

Returns: predicted crop, disease name, and ensemble confidence.
"""
import torch
import numpy as np
import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.utils import get_config, get_transform, get_device
from core.dataset import load_classes
from models.densenet_121 import get_densenet121


def load_model(model_path, num_classes, device):
    model = get_densenet121(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict(path_to_image: str):
    config_path = 'densenet121/config.yaml'
    config = get_config(config_path)
    device = get_device()
    num_classes = config['num_classes']
    k_fold = config['k_fold']
    backbone_dir = config['backbone']['dir']

    models = [
        load_model(os.path.join(backbone_dir, f'backbone_fold_{i + 1}.pth'), num_classes, device)
        for i in range(k_fold)
    ]

    classes = load_classes(config_path)

    transform = get_transform('test', config_path)
    img = Image.open(path_to_image).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # Soft-voting across backbone ensemble
    with torch.no_grad():
        outputs = [m(tensor) for m in models]
    avg_output = torch.stack(outputs, dim=0).mean(dim=0)  # (1, num_classes)
    proba = torch.softmax(avg_output, dim=1).cpu().numpy()

    predicted_index = proba.argmax()
    confidence = proba[0][predicted_index]
    predicted_label = classes[predicted_index]
    crop, disease = predicted_label.split('__', 1)
    return crop, disease, confidence


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else './images.jpg'
    crop, disease, confidence = predict(image_path)
    print(f"Predicted Crop   : {crop}")
    print(f"Predicted Disease: {disease}")
    print(f"Confidence       : {confidence:.4f}")
