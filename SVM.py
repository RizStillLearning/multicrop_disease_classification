import torch
import torch.nn as nn
import os
from PIL import Image
from dataset import load_dataset, split_dataset, build_dataloader
from model import build_model, load_model, extract_features
from utils import get_transform, get_config, get_device
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

device = get_device()
config = get_config()

model_name = config['model_name']
model = build_model(num_classes=config['num_classes'])
model.to(device)
final_model_path = os.path.join(config['final_model_dir'], config['final_model_name'])
crop_disease_classes = load_model(final_model_path, model)
model.classifier[1] = nn.Identity() 
model.eval()

print("Loading dataset...")
df, _ = load_dataset(config['dataset_dir'])
train_df, val_df, test_df = split_dataset(df)

train_loader = build_dataloader(train_df, mode='train')
val_loader = build_dataloader(val_df, mode='val')
test_loader = build_dataloader(test_df, mode='test')

print("Extracting features from training data...")
train_features, train_labels = extract_features(model, train_loader, device)

print("Extracting features from validation data...")
val_features, val_labels = extract_features(model, val_loader, device)

print("Extracting features from test data...")
test_features, test_labels = extract_features(model, test_loader, device)

print("Training SVM model...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(train_features, train_labels)

print("Evaluating SVM model on validation data...")
val_predictions = svm_model.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("Evaluating SVM model on test data...")
test_predictions = svm_model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")
