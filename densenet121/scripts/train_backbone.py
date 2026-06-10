import torch
import os
import gc
import copy
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.dataset import load_dataset, build_dataloader
from core.utils import (
    get_config, get_device, load_checkpoint, save_checkpoint,
    save_current_fold, load_current_fold, seed_everything
)
from core.train import train_model, validate_model, save_classification_report, write_training_log
from models.densenet_121 import get_densenet121


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_model(num_classes):
    return get_densenet121(num_classes=num_classes, pretrained=True)


def save_model(model_dir, model_name, model):
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, model_name)
    torch.save({'model_state_dict': model.state_dict()}, save_path)


def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=get_device(), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seed_everything()
    config_name = 'densenet121/config.yaml'
    config = get_config(config_name)

    print("Loading dataset...")
    df, classes = load_dataset(config_name)

    print("Splitting dataset into train, validation, and test data...")
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['crop_disease']
    )

    print("Building test dataloader...")
    test_dataloader = build_dataloader(test_df, 'test', config_name)

    num_classes = len(classes)
    k_fold = config['k_fold']
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    current_fold = 0

    fold_results = pd.DataFrame({
        'Fold': pd.Series(dtype='int8'),
        'Best Validation Loss': pd.Series(dtype='float'),
        'Best Validation Accuracy': pd.Series(dtype='float'),
    })

    fold_results_config = config['fold_results']
    fold_results_path = os.path.join(
        fold_results_config['dir'], fold_results_config['backbone']
    )

    if os.path.exists(fold_results_path):
        fold_results = load_current_fold(fold_results_path)
        current_fold = len(fold_results)
        print(f"Resuming from fold {current_fold + 1}")
    else:
        print("No fold results found. Starting from fold 1.")

    device = get_device()
    y_numpy = np.array(train_val_df['crop_disease'].values)
    backbone_models = []
    backbone_config = config['backbone']
    backbone_dir = backbone_config['dir']

    for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_numpy)), y_numpy)):
        if i < current_fold:
            continue

        print(f"Fold {i + 1}/{k_fold}")
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_dataloader = build_dataloader(train_df, mode='train', config_path=config_name)
        val_dataloader = build_dataloader(val_df, mode='val', config_path=config_name)

        model = build_model(num_classes=num_classes)
        model.to(device)

        cur_epoch = 1
        best_val_loss = float('inf')
        best_val_acc = float('-inf')
        best_model_state = build_model(num_classes=num_classes)
        best_model_state.to(device)

        checkpoint = config['checkpoint']
        checkpoint_path = os.path.join(checkpoint['dir'], checkpoint['name'])

        crop_disease_labels = train_df['crop_disease'].values
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(crop_disease_labels),
            y=crop_disease_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        num_epoch = config['num_epoch']

        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            cur_epoch, best_val_loss, best_val_acc = load_checkpoint(
                checkpoint_path, model, best_model_state, optimizer
            )
            print(
                f"Resuming from epoch {cur_epoch} | "
                f"best val loss {best_val_loss:.4f} | "
                f"best val acc {best_val_acc:.4f}"
            )
        else:
            print("No checkpoint found. Starting training from scratch.")

        print("Training model...")
        patience = 10
        epochs_without_improvement = 0

        for epoch in range(cur_epoch, num_epoch + 1):
            train_loss = train_model(train_dataloader, model, device, optimizer, criterion)
            scheduler.step()
            val_loss, val_acc = validate_model(val_dataloader, model, device, criterion)
            print(
                f"Epoch [{epoch}/{num_epoch}] "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}"
            )

            write_training_log(config_name, epoch, train_loss, val_loss, val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model)
                epochs_without_improvement = 0
                print("Best model updated.")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{patience} epochs.")

            save_checkpoint(
                checkpoint['dir'], checkpoint['name'],
                best_model_state, optimizer, epoch + 1, best_val_loss, best_val_acc
            )
            print(f"Checkpoint saved. Best val loss: {best_val_loss:.4f} | Best val acc: {best_val_acc:.4f}")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

            gc.collect()
            torch.cuda.empty_cache()

        fold_results.loc[i] = [i + 1, f"{best_val_loss:.4f}", f"{best_val_acc:.4f}"]
        save_current_fold(fold_results_config['dir'], fold_results_config['backbone'], fold_results)
        print(f"Fold {i + 1} done. Best val loss: {best_val_loss:.4f} | Best val acc: {best_val_acc:.4f}")

        if i < k_fold - 1:
            os.remove(checkpoint_path)

        backbone_name = f'backbone_fold_{i + 1}.pth'
        save_model(backbone_dir, backbone_name, best_model_state)

    # ── Ensemble evaluation ────────────────────────────────────────────────────
    print("Evaluating ensemble models on test set...")
    for i in range(k_fold):
        backbone_name = f'backbone_fold_{i + 1}.pth'
        backbone_path = os.path.join(backbone_dir, backbone_name)
        backbone = build_model(num_classes)
        load_model(backbone_path, backbone)
        backbone.to(device)
        backbone_models.append(backbone)

    for m in backbone_models:
        m.eval()

    y_true_ensemble = []
    y_pred_ensemble = []

    loop = tqdm(test_dataloader, desc='Ensemble evaluation', leave=False)
    with torch.no_grad():
        for image, label in loop:
            image, label = image.to(device), label.to(device)
            outputs = [m(image) for m in backbone_models]
            outputs = torch.stack(outputs).mean(dim=0)
            _, pred = torch.max(outputs, dim=1)
            y_true_ensemble.extend(label.cpu().numpy())
            y_pred_ensemble.extend(pred.cpu().numpy())

    y_true = np.array(y_true_ensemble)
    y_pred = np.array(y_pred_ensemble)

    accuracy = (y_true == y_pred).sum() / len(y_true)
    print(f"Ensemble Test Accuracy: {accuracy:.4f}")

    report_config = config['classification_report']
    save_classification_report(
        y_true, y_pred, classes,
        report_config['dir'], report_config['backbone']
    )

    plot_dir = config['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'backbone_confusion_matrix.png')

    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('DenseNet-121 Ensemble Confusion Matrix')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")


if __name__ == '__main__':
    main()
