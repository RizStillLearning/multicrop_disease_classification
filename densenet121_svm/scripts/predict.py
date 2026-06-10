"""
predict.py — DenseNet-121 + SVM inference

Usage (from project root):
    python densenet121_svm/scripts/predict.py <path_to_image>

Returns: predicted crop, disease name, and ensemble confidence.
"""
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.utils import get_config, get_transform, get_device
from core.dataset import load_classes
from models.densenet_121 import get_densenet121


class DenseNet121FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features

    def forward(self, x):
        import torch.nn.functional as F
        out = self.features(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)


def load_backbone(model_path, num_classes, device):
    backbone = get_densenet121(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    extractor = DenseNet121FeatureExtractor(backbone)
    extractor.to(device)
    extractor.eval()
    return extractor


def predict(path_to_image: str):
    config_path = 'densenet121_svm/config.yaml'
    config = get_config(config_path)
    device = get_device()
    num_classes = config['num_classes']
    k_fold = config['k_fold']
    backbone_dir = config['backbone']['dir']
    classifier_dir = config['classifier']['dir']

    # Load feature extractors
    feature_extractors = [
        load_backbone(os.path.join(backbone_dir, f'backbone_fold_{i + 1}.pth'), num_classes, device)
        for i in range(k_fold)
    ]

    # Load SVM models + scalers
    svm_models = [
        joblib.load(os.path.join(classifier_dir, f'svm_fold_{i + 1}.joblib'))
        for i in range(k_fold)
    ]
    scalers = [
        joblib.load(os.path.join(classifier_dir, f'scaler_fold_{i + 1}.joblib'))
        for i in range(k_fold)
    ]

    classes = load_classes(config_path)

    transform = get_transform('test', config_path)
    img = Image.open(path_to_image).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # Extract and average features across folds
    with torch.no_grad():
        fold_feats = [extractor(tensor) for extractor in feature_extractors]
    feature = torch.stack(fold_feats, dim=0).mean(dim=0).cpu().numpy()   # (1, 1024)

    # Ensemble SVM soft-voting
    predictions = []
    for svm, scaler in zip(svm_models, scalers):
        scaled = scaler.transform(feature)
        predictions.append(svm.predict_proba(scaled))

    avg_proba = np.stack(predictions, axis=0).mean(axis=0)
    predicted_index = avg_proba.argmax()
    confidence = avg_proba[0][predicted_index]
    predicted_label = classes[predicted_index]
    crop, disease = predicted_label.split('__', 1)
    return crop, disease, confidence


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else './images.jpg'
    crop, disease, confidence = predict(image_path)
    print(f"Predicted Crop   : {crop}")
    print(f"Predicted Disease: {disease}")
    print(f"Confidence       : {confidence:.4f}")
