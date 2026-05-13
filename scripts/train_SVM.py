import torch.nn as nn
import numpy as np
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dataset import load_dataset, build_dataloader
from core.model import build_model, load_model
from core.utils import get_config, get_device, save_current_fold, seed_everything
from core.train import extract_features, save_classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    seed_everything()  # Set random seed for reproducibility
    device = get_device()
    config = get_config()

    k_fold = config['k_fold']
    backbone_models = []
    for i in range(k_fold):
        model = build_model(num_classes=config['num_classes'])
        model.to(device)
        backbone_path = os.path.join(config['backbone_dir'], f'backbone_fold_{i+1}.pth')
        crop_disease_classes = load_model(backbone_path, model)
        model.classifier[1] = nn.Identity() 
        model.eval()
        backbone_models.append(model)

    print("Loading dataset...")
    df, crop_disease_classes = load_dataset(config['dataset_dir'])
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=50, stratify=df['crop_disease'])
    
    skf = StratifiedKFold(n_splits=config['k_fold'], shuffle=True, random_state=50)
    y_numpy = np.array(train_val_df['crop_disease'].values)

    training_log_dir = config['training_log_dir']
    fold_results_name = config['svm_fold_results_name']
    fold_results = pd.DataFrame({
        'Fold': pd.Series(dtype='int8'),
        'Validation Accuracy': pd.Series(dtype='float')
    })

    cur_fold = 0
    if os.path.exists(config['classifier_dir']):
        cur_fold = len([f for f in os.listdir(config['classifier_dir']) if f.startswith('svm_fold_')])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_numpy)), y_numpy)):
        if fold < cur_fold:
            continue

        print(f"Processing fold {fold + 1}/{config['k_fold']}...")
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_dataloader = build_dataloader(train_df, mode='train')
        val_dataloader = build_dataloader(val_df, mode='val')

        print("Extracting features for training data...")
        train_features, train_labels = extract_features(backbone_models, train_dataloader, device)

        print("Extracting features for validation data...")
        val_features, val_labels = extract_features(backbone_models, val_dataloader, device)

        print("Training SVM model...")
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(train_features, train_labels)

        print("Evaluating SVM model on validation data...")
        val_predictions = svm_model.predict(val_features)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        fold_results.loc[fold] = [fold + 1, f"{val_accuracy:.4f}"]
        save_current_fold(training_log_dir, fold_results, fold_name=fold_results_name)

        svm_model_path = os.path.join(config['classifier_dir'], f'svm_fold_{fold+1}.joblib')
        os.makedirs(config['classifier_dir'], exist_ok=True)
        joblib.dump(svm_model, svm_model_path)
        print(f"SVM model saved to {svm_model_path}")

    print("Evaluating final SVM model on test data...")
    test_dataloader = build_dataloader(test_df, mode='test')
    test_features, test_labels = extract_features(backbone_models, test_dataloader, device)

    svm_models = []
    scalers = []
    for fold in range(config['k_fold']):
        svm_model_path = os.path.join(config['classifier_dir'], f'svm_fold_{fold+1}.joblib')
        svm_models.append(joblib.load(svm_model_path))

    test_predictions = []
    for svm_model in svm_models:
        pred = svm_model.predict_proba(test_features)
        test_predictions.append(pred)

    test_predictions = np.stack(test_predictions, axis=0)
    test_predictions = np.mean(test_predictions, axis=0)
    test_predictions = test_predictions.argmax(axis=1)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    plot_dir = config['plot_dir']
    plot_name = 'svm_confusion_matrix.png'
    plot_path = os.path.join(plot_dir, plot_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Generating confusing matrix...")
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=crop_disease_classes, yticklabels=crop_disease_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(plot_path)
    plt.close()
    print("Confusion matrix saved.")

    print("Classification Report on Test Data:")
    report = classification_report(test_labels, test_predictions, target_names=crop_disease_classes)
    print(report)

    save_classification_report(test_labels, test_predictions, crop_disease_classes, config['classification_report_dir'], config['svm_classification_report_name'])

    del model

if __name__ == '__main__':
    main()