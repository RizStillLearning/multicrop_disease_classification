import torch.nn as nn
import numpy as np
import os
import joblib
import pandas as pd
from dataset import load_dataset, build_dataloader
from model import build_model, load_model, extract_features
from utils import get_config, get_device, save_current_fold, seed_everything
from train_backbone import save_classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def main():
    seed_everything()  # Set random seed for reproducibility
    device = get_device()
    config = get_config()

    model = build_model(num_classes=config['num_classes'])
    model.to(device)
    backbone_path = os.path.join(config['backbone_dir'], config['backbone_name'])
    crop_disease_classes = load_model(backbone_path, model)
    model.classifier[1] = nn.Identity() 
    model.eval()

    print("Loading dataset...")
    df, _ = load_dataset(config['dataset_dir'])
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['crop_disease'])
    
    skf = StratifiedKFold(n_splits=config['k_fold'], shuffle=True, random_state=42)
    y_numpy = np.array(train_val_df['crop_disease'].values)

    training_log_dir = config['training_log_dir']
    fold_results_name = config['svm_fold_results_name']
    fold_results_path = os.path.join(training_log_dir, fold_results_name)
    fold_results = pd.DataFrame({
        'fold': pd.Series(dtype='int8'),
        'val_accuracy': pd.Series(dtype='float')
    })

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_numpy)), y_numpy)):
        print(f"Processing fold {fold + 1}/{config['k_fold']}...")
        train_df = train_val_df.iloc[train_idx]
        test_df = train_val_df.iloc[val_idx]

        train_dataloader = build_dataloader(train_df, mode='train')
        val_dataloader = build_dataloader(test_df, mode='val')

        print("Extracting features for training data...")
        train_features, train_labels = extract_features(model, train_dataloader, device)

        print("Extracting features for validation data...")
        val_features, val_labels = extract_features(model, val_dataloader, device)

        print("Standardizing features...")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        print("Training SVM model...")
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(train_features, train_labels)

        print("Evaluating SVM model on validation data...")
        val_predictions = svm_model.predict(val_features)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        fold_results.loc[fold] = [fold + 1, f"{val_accuracy:.4f}"]
        save_current_fold(training_log_dir, fold_results, fold_name=fold_results_name)

    print("Evaluating final SVM model on test data...")
    test_dataloader = build_dataloader(test_df, mode='test')
    test_features, test_labels = extract_features(model, test_dataloader, device)
    test_features = scaler.transform(test_features)
    test_predictions = svm_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report on Test Data:")
    report = classification_report(test_labels, test_predictions, target_names=crop_disease_classes)
    print(report)

    save_classification_report(test_labels, test_predictions, crop_disease_classes, config['classification_report_dir'], 'svm_classification_report.json')

    svm_model_path = os.path.join(config['classifier_dir'], config['classifier_name'])
    os.makedirs(config['classifier_dir'], exist_ok=True)
    joblib.dump(svm_model, svm_model_path)
    print(f"SVM model saved to {svm_model_path}")

if __name__ == '__main__':
    main()