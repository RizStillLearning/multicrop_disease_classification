import torch
import torch.nn as nn
import os
import joblib
from PIL import Image
from dataset import load_dataset, split_dataset, build_dataloader
from model import build_model, load_model, extract_features
from utils import get_transform, get_config, get_device
from train import save_classification_report
from sklearn.model_selection import train_test_split
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
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['crop_disease'])

train_loader = build_dataloader(train_df, mode='train')
test_loader = build_dataloader(test_df, mode='test')

print("Extracting features from training data...")
train_features, train_labels = extract_features(model, train_loader, device)

print("Extracting features from test data...")
test_features, test_labels = extract_features(model, test_loader, device)

print("Standardizing features...")
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

print("Training SVM model...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(train_features, train_labels)

print("Evaluating SVM model on test data...")
test_predictions = svm_model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report on Test Data:")
report = classification_report(test_labels, test_predictions, target_names=crop_disease_classes, output_dict=True)
print(classification_report(test_labels, test_predictions, target_names=crop_disease_classes))

save_classification_report(test_labels, test_predictions, crop_disease_classes, config['classification_report_dir'], 'svm_classification_report.json')

svm_model_path = os.path.join(config['final_model_dir'], 'svm_model.joblib')
os.makedirs(config['final_model_dir'], exist_ok=True)
joblib.dump(svm_model, svm_model_path)
print(f"SVM model saved to {svm_model_path}")