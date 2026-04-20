import torch
import os
import gc
import copy
import json
import csv
import numpy as np
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from dataset import load_dataset, split_dataset, build_dataloader
from utils import get_config, get_device, load_checkpoint, save_checkpoint
from model import build_model, save_model

def validate_model(val_loader, model, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, crop_disease_labels in val_loader:
            images, crop_disease_labels = images.to(device), crop_disease_labels.to(device)
            output = model(images)

            loss = criterion(output, crop_disease_labels)
            total_loss += loss.item()

            _, predicted = torch.max(output, dim=1)
            total += crop_disease_labels.size(0)
            correct += (predicted == crop_disease_labels).sum().item()

    loss = total_loss / len(val_loader)
    acc = correct / total
    return loss, acc    

def train_model(train_loader, model, device, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, crop_disease_labels in train_loader:
        images, crop_disease_labels = images.to(device), crop_disease_labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, crop_disease_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss = total_loss / len(train_loader)
    return loss

def evaluate_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, crop_disease_labels in test_loader:
            images, crop_disease_labels = images.to(device), crop_disease_labels.to(device)
            output = model(images)

            _, predicted = torch.max(output, dim=1)
            total += crop_disease_labels.size(0)
            correct += (predicted == crop_disease_labels).sum().item()

    acc = correct / total
    return acc

def get_labels(test_loader, model, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, crop_disease_labels in test_loader:
            images, crop_disease_labels = images.to(device), crop_disease_labels.to(device)
            output = model(images)

            _, predicted = torch.max(output, dim=1)

            y_true.extend(crop_disease_labels.detach().cpu().numpy())
            y_pred.extend(predicted.detach().cpu().numpy())

    return np.array(y_true), np.array(y_pred)

def get_metrics_per_class(y_true, y_pred, target_names):
    tp = np.zeros(len(target_names))
    fp = np.zeros(len(target_names))
    fn = np.zeros(len(target_names))
    tn = np.zeros(len(target_names))

    for i in range(len(target_names)):
        tp[i] = np.sum((y_true == i) & (y_pred == i))
        fp[i] = np.sum((y_true != i) & (y_pred == i))
        fn[i] = np.sum((y_true == i) & (y_pred != i))
        tn[i] = np.sum((y_true != i) & (y_pred != i))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }

    return metrics

def write_training_log(epoch, train_loss, val_loss, val_acc):
    config = get_config()
    file_name = config['training_log_name']
    log_dir = config['training_log_dir']
    file_path = os.path.join(log_dir, file_name)
    os.makedirs(log_dir, exist_ok=True)
    mode = 'w' if epoch == 1 else 'a'
    with open(file_path, mode, newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if epoch == 1:
            csvwriter.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy'])
        csvwriter.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

def save_classification_report(y_true, y_pred, target_names, report_dir, file_name):
    file_path = os.path.join(report_dir, file_name)
    os.makedirs(report_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    with open(file_path, 'w') as f:
        json.dump(report, f, indent=4)


def main():
    config = get_config()
    model_name = config['model_name']

    print("Loading dataset...")
    df, crop_disease_classes = load_dataset(config['dataset_dir'])

    print("Splitting dataset into train, validation, and test data...")
    train_df, val_df, test_df = split_dataset(df)

    print("Building dataloader...")
    train_dataloader = build_dataloader(train_df, 'train')
    val_dataloader = build_dataloader(val_df, 'val')
    test_dataloader = build_dataloader(test_df, 'test')

    num_crop_disease_classes = len(crop_disease_classes)

    checkpoint_name = config['checkpoint_name']
    checkpoint_dir = config['checkpoint_dir']
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    model = build_model(num_classes=num_crop_disease_classes)
    device = get_device()
    model.to(device)

    cur_epoch = 1
    best_val_loss = float('inf')
    best_model_state = build_model(num_classes=num_crop_disease_classes)
    best_model_state.to(device)

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
        print(f"Current epoch: {cur_epoch}")

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

    print("Evaluating model...")
    acc = evaluate_model(test_dataloader, best_model_state, device)
    print(f"Test accuracy: {acc:.4f}")
    
    final_model_dir = config['final_model_dir']
    final_model_name = config['final_model_name']
    save_model(final_model_dir, final_model_name, best_model_state, crop_disease_classes=crop_disease_classes)

    y_true, y_pred = get_labels(test_dataloader, best_model_state, device)
    classification_report_dir = config['classification_report_dir']
    classification_report_name = config['classification_report_name']
    save_classification_report(y_true, y_pred, crop_disease_classes, classification_report_dir, classification_report_name)

    del model

if __name__ == '__main__':
    main()