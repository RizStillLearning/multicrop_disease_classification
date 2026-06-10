import torch
import torch.nn as nn
import numpy as np
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.dataset import load_dataset, load_classes, build_dataloader
from core.utils import get_config, get_device, save_current_fold, seed_everything
from core.train import save_classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, zero_one_loss, confusion_matrix
from models.densenet_121 import get_cbam_densenet121


# ── Feature extractor helpers ─────────────────────────────────────────────────

class DenseNet121CBAMFeatureExtractor(nn.Module):
    '''
    Wraps a trained DenseNet-121 + CBAM backbone and returns 1024-d feature vectors.

    After the features block (which now contains DenseBlockWithCBAM wrappers),
    the output is passed through BN+ReLU (norm5), then global average pooling,
    and flattened.  The classifier head is bypassed.
    '''

    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features       # DenseBlocks (w/ CBAM) + transitions + norm5

    def forward(self, x):
        out = self.features(x)
        out = torch.nn.functional.relu(out, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)             # (B, 1024)
        return out


def load_model(model_path, model):
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])


def extract_features_from_ensemble(feature_extractors, dataloader, device):
    '''
    Run every image through all feature-extractor folds,
    average the 1024-d vectors, return (features, labels).
    '''
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            fold_feats = []
            for extractor in feature_extractors:
                extractor.eval()
                fold_feats.append(extractor(images))

            avg_feats = torch.stack(fold_feats, dim=0).mean(dim=0)
            all_features.append(avg_feats.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seed_everything()
    device = get_device()

    config_path = 'densenet121_cbam_svm/config.yaml'
    config = get_config(config_path)

    k_fold = config['k_fold']
    backbone_dir = config['backbone']['dir']
    num_classes = config['num_classes']

    # Load all backbone folds (with CBAM) and wrap as feature extractors
    feature_extractors = []
    for i in range(k_fold):
        backbone_path = os.path.join(backbone_dir, f'backbone_fold_{i + 1}.pth')
        backbone = get_cbam_densenet121(num_classes=num_classes, pretrained=False)
        load_model(backbone_path, backbone)
        extractor = DenseNet121CBAMFeatureExtractor(backbone)
        extractor.to(device)
        extractor.eval()
        feature_extractors.append(extractor)

    print("Loading dataset...")
    df, classes = load_dataset(config_path)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=50, stratify=df['crop_disease']
    )

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=50)
    y_numpy = np.array(train_val_df['crop_disease'].values)

    fold_results_config = config['fold_results']
    fold_results = pd.DataFrame({
        'Fold': pd.Series(dtype='int8'),
        'Validation Loss': pd.Series(dtype='float'),
        'Validation Accuracy': pd.Series(dtype='float'),
    })

    classifier_dir = config['classifier']['dir']
    cur_fold = 0
    if os.path.exists(classifier_dir):
        cur_fold = len([f for f in os.listdir(classifier_dir) if f.startswith('svm_fold_')])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_numpy)), y_numpy)):
        if fold < cur_fold:
            continue

        print(f"Processing fold {fold + 1}/{k_fold}...")
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_dataloader = build_dataloader(train_df, mode='train', config_path=config_path)
        val_dataloader = build_dataloader(val_df, mode='val', config_path=config_path)

        print("Extracting features for training data...")
        train_features, train_labels = extract_features_from_ensemble(
            feature_extractors, train_dataloader, device
        )

        print("Extracting features for validation data...")
        val_features, val_labels = extract_features_from_ensemble(
            feature_extractors, val_dataloader, device
        )

        # Normalize features with StandardScaler
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        print("Training SVM classifier...")
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(train_features, train_labels)

        print("Evaluating SVM on validation data...")
        val_predictions = svm_model.predict(val_features)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_loss = zero_one_loss(val_labels, val_predictions)
        print(f"Validation Accuracy: {val_accuracy:.4f}  Validation Loss: {val_loss:.4f}")

        fold_results.loc[fold] = [fold + 1, f"{val_loss:.4f}", f"{val_accuracy:.4f}"]
        save_current_fold(
            fold_results_config['dir'], fold_results_config['classifier'], fold_results
        )

        os.makedirs(classifier_dir, exist_ok=True)
        svm_model_path = os.path.join(classifier_dir, f'svm_fold_{fold + 1}.joblib')
        scaler_path = os.path.join(classifier_dir, f'scaler_fold_{fold + 1}.joblib')
        joblib.dump(svm_model, svm_model_path)
        joblib.dump(scaler, scaler_path)
        print(f"SVM model saved to {svm_model_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("Evaluating final ensemble SVM on test data...")
    test_dataloader = build_dataloader(test_df, mode='test', config_path=config_path)
    test_features, test_labels = extract_features_from_ensemble(
        feature_extractors, test_dataloader, device
    )

    svm_models = []
    scalers = []
    for fold in range(k_fold):
        svm_models.append(joblib.load(os.path.join(classifier_dir, f'svm_fold_{fold + 1}.joblib')))
        scalers.append(joblib.load(os.path.join(classifier_dir, f'scaler_fold_{fold + 1}.joblib')))

    test_predictions = []
    for svm_model, scaler in zip(svm_models, scalers):
        scaled_features = scaler.transform(test_features)
        pred = svm_model.predict_proba(scaled_features)
        test_predictions.append(pred)

    # Soft-voting ensemble over SVM folds
    test_predictions = np.stack(test_predictions, axis=0).mean(axis=0)
    test_predictions = test_predictions.argmax(axis=1)

    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_loss = zero_one_loss(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}  Test Loss: {test_loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, target_names=classes))

    report_config = config['classification_report']
    save_classification_report(
        test_labels, test_predictions, classes,
        report_config['dir'], report_config['svm']
    )

    plot_dir = config['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'svm_confusion_matrix.png')

    print("Generating confusion matrix...")
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('DenseNet-121 + CBAM + SVM Confusion Matrix')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")


if __name__ == '__main__':
    main()
