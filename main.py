import torch
import os
import gc
import copy
import numpy as np
import torch.optim as optim
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from dataset import load_dataset, split_dataset, build_dataloader
from utils import get_config, get_device, load_checkpoint, save_checkpoint
from model import build_model, save_model
from train import train_model, validate_model, evaluate_model, write_training_log, get_labels, save_classification_report

def main():
    config = get_config()
    model_name = config['model_name']

    print("Loading dataset...")
    df, crop_classes, crop_disease_classes = load_dataset(config['dataset_dir'])

    print("Splitting dataset into train, validation, and test data...")
    train_df, val_df, test_df = split_dataset(df)

    print("Building dataloader...")
    train_dataloader = build_dataloader(train_df, 'train')
    val_dataloader = build_dataloader(val_df, 'val')
    test_dataloader = build_dataloader(test_df, 'test')

    num_crop_classes = len(crop_classes)
    num_crop_disease_classes = len(crop_disease_classes)

    checkpoint_name = config['checkpoint_name']
    checkpoint_path = os.path.join(model_name, checkpoint_name)

    model = build_model(model_name, num_crop_classes, num_crop_disease_classes)
    device = get_device()
    model.to(device)

    cur_epoch = 1
    best_val_loss = float('inf')
    best_model_state = build_model(model_name, num_crop_classes, num_crop_disease_classes)
    best_model_state.to(device)

    crop_labels = train_df['crop'].values
    class_weights_crop = compute_class_weight(class_weight='balanced', classes=np.unique(crop_labels), y=crop_labels)
    class_weights_crop = torch.tensor(class_weights_crop, dtype=torch.float).to(device)

    crop_disease_labels = train_df['crop_disease'].values
    class_weights_crop_disease = compute_class_weight(class_weight='balanced', classes=np.unique(crop_disease_labels), y=crop_disease_labels)
    class_weights_crop_disease = torch.tensor(class_weights_crop_disease, dtype=torch.float).to(device)

    criterion_crop = torch.nn.CrossEntropyLoss(weight=class_weights_crop)
    criterion_crop_disease = torch.nn.CrossEntropyLoss(weight=class_weights_crop_disease)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    num_epoch = config['num_epoch']

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        cur_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, best_model_state, optimizer)
        print(f"Current epoch: {cur_epoch}")

    print("Training model...")
    for epoch in range(cur_epoch, num_epoch+1):
        train_loss = train_model(train_dataloader, model, device, optimizer, criterion_crop, criterion_crop_disease)
        val_loss, val_acc = validate_model(val_dataloader, model, device, criterion_crop, criterion_crop_disease)
        print(f"Epoch [{epoch}/{num_epoch}]\nTrain Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        write_training_log(model_name, epoch, train_loss, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model)
            print("Best model saved.")
            
        save_checkpoint(model_name, checkpoint_name, best_model_state, optimizer, epoch + 1, best_val_loss)
        print(f"Current best validation loss: {best_val_loss:.4f}")
        print("Checkpoint saved.")

        gc.collect()
        torch.cuda.empty_cache()

    print("Evaluating model...")
    acc = evaluate_model(test_dataloader, best_model_state, device)
    print(f"Test accuracy: {acc:.4f}")
    
    save_model(model_name, best_model_state, crop_classes=crop_classes, crop_disease_classes=crop_disease_classes)

    y1_true, y1_pred, y2_true, y2_pred = get_labels(test_dataloader, best_model_state, device)
    save_classification_report(model_name, y1_true, y1_pred, crop_classes, 'crop_classification_report.json')
    save_classification_report(model_name, y2_true, y2_pred, crop_disease_classes, 'crop_disease_classification_report.json')

    del model

if __name__ == '__main__':
    main()