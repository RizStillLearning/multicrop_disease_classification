import os
import torch
import json
import csv
import numpy as np
from utils import get_config
from sklearn.metrics import classification_report

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