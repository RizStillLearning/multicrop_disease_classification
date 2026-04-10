import torch
import os
import csv
import json
import numpy as np
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

def write_training_log(model_name, epoch, train_loss, val_loss, val_acc, output_name='training_log.csv'):
    file_path = os.path.join(model_name, output_name)
    os.makedirs(model_name, exist_ok=True)
    mode = 'w' if epoch == 1 else 'a'
    with open(file_path, mode, newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if epoch == 1:
            csvwriter.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy'])
        csvwriter.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

def save_classification_report(model_name, y_true, y_pred, target_names, file_name):
    os.makedirs(model_name, exist_ok=True)
    file_path = os.path.join(model_name, file_name)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    with open(file_path, 'w') as f:
        json.dump(report, f, indent=4)