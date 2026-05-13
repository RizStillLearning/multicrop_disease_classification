import torch
import os
import gc
import copy
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dataset import load_dataset, build_dataloader
from core.utils import get_config, get_device, load_checkpoint, save_checkpoint, save_current_fold, load_current_fold, seed_everything
from core.model import build_model, save_model, load_model
from core.train import train_model, validate_model, save_classification_report, write_training_log

def main():
    seed_everything()  # Set random seed for reproducibility
    config = get_config()

    print("Loading dataset...")    
    
    df, crop_disease_classes = load_dataset(config['dataset_dir'])

    print("Splitting dataset into train, validation, and test data...")
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['crop_disease'])

    print("Building dataloader...")
    test_dataloader = build_dataloader(test_df, 'test')

    num_crop_disease_classes = len(crop_disease_classes)

    k_fold = config['k_fold']
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    current_fold = 0
    fold_results = pd.DataFrame({
        'Fold': pd.Series(dtype='int8'),
        'Best Validation Loss': pd.Series(dtype='float')
    })

    # Load current fold to resume training
    training_log_dir = config['training_log_dir']
    fold_results_name = config['backbone_fold_results_name']
    fold_results_path = os.path.join(training_log_dir, fold_results_name)

    if os.path.exists(fold_results_path):
        fold_results = load_current_fold(training_log_dir, fold_results_name)
        current_fold = len(fold_results)
        print(f"Resuming from fold {current_fold + 1}")
    else:
        print("No fold results found. Starting from fold 1.")

    y_numpy = np.array(train_val_df['crop_disease'].values)
    backbone_models = []

    for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_numpy)), y_numpy)):
        if i < current_fold:
            continue

        print(f"Fold {i + 1}/{k_fold}")

        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_dataloader = build_dataloader(train_df, mode='train')
        val_dataloader = build_dataloader(val_df, mode='val')

        model = build_model(num_classes=num_crop_disease_classes)
        device = get_device()
        model.to(device)

        cur_epoch = 1
        best_val_loss = float('inf')
        best_model_state = build_model(num_classes=num_crop_disease_classes)
        best_model_state.to(device)

        checkpoint_dir = config['checkpoint_dir']
        checkpoint_name = config['checkpoint_name']
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        crop_disease_labels = train_df['crop_disease'].values
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(crop_disease_labels), y=crop_disease_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        num_epoch = config['num_epoch']

        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            cur_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, best_model_state, optimizer)
            print(f"Resuming training from epoch {cur_epoch} with best validation loss {best_val_loss:.4f}")
        else:
            print("No checkpoint found. Starting training from scratch.")

        print("Training model...")
        for epoch in range(cur_epoch, num_epoch+1):
            train_loss = train_model(train_dataloader, model, device, optimizer, criterion)
            scheduler.step()
            val_loss, val_acc = validate_model(val_dataloader, model, device, criterion)
            print(f"Epoch [{epoch}/{num_epoch}]\nTrain Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

            write_training_log(epoch, train_loss, val_loss, val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model)
                print("Best model saved.")
                
            save_checkpoint(checkpoint_dir, checkpoint_name, best_model_state, optimizer, epoch + 1, best_val_loss)
            print(f"Current best validation loss: {best_val_loss:.4f}")
            print("Checkpoint saved.")

            gc.collect()
            torch.cuda.empty_cache()

        fold_results.loc[i] = [i + 1, f"{best_val_loss:.4f}"]
        save_current_fold(training_log_dir, fold_results, fold_name=fold_results_name)
        print(f"Fold {i + 1} completed. Best validation loss: {best_val_loss:.4f}")

        if i < k_fold - 1:
            os.remove(checkpoint_path) # Clean up checkpoint after each fold

        backbone_dir = config['backbone_dir']
        backbone_name = f'backbone_fold_{i+1}.pth'
        save_model(backbone_dir, backbone_name, best_model_state, crop_disease_classes=crop_disease_classes)

    print("Evaluating ensemble models...")
    for i in range(k_fold):
        backbone_dir = config['backbone_dir']
        backbone_name = f'backbone_fold_{i+1}.pth'
        backbone_path = os.path.join(backbone_dir, backbone_name)
        backbone = build_model(num_crop_disease_classes)
        load_model(backbone_path, backbone)
        backbone.to(device)
        backbone_models.append(backbone)

    for model in backbone_models:
        model.eval()

    y_true_ensemble = []
    y_pred_ensemble = []

    with torch.no_grad():
        for image, label in test_dataloader:
            image, label = image.to(device), label.to(device)
            outputs = []
            
            for model in backbone_models:
                outputs.append(model(image))
            
            outputs = torch.stack(outputs).mean(dim=0)
            _, pred = torch.max(outputs, dim=1)
            
            y_true_ensemble.extend(label.cpu().numpy())
            y_pred_ensemble.extend(pred.cpu().numpy())
            
    y_true = np.array(y_true_ensemble)
    y_pred = np.array(y_pred_ensemble)
    
    accuracy = (y_true == y_pred).sum() / len(y_true)
    print(f"Ensemble Test Accuracy: {accuracy:.4f}")
    classification_report_dir = config['classification_report_dir']
    classification_report_name = config['backbone_classification_report_name']
    save_classification_report(y_true, y_pred, crop_disease_classes, classification_report_dir, classification_report_name)

    plot_dir = config['plot_dir']
    plot_name = 'backbone_confusion_matrix.png'
    plot_path = os.path.join(plot_dir, plot_name)

    os.makedirs(plot_dir, exist_ok=True)

    print("Generating confusing matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=crop_disease_classes, yticklabels=crop_disease_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(plot_path)
    plt.close()
    print("Confusion matrix saved.")

    del model

if __name__ == '__main__':
    main()