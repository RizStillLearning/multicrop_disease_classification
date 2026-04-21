import joblib
import torch.nn as nn
import os
from PIL import Image
from model import build_model, load_model
from utils import get_transform, get_config

config = get_config()
model_name = config['model_name']
model = build_model(num_classes=config['num_classes'])
final_model_path = os.path.join(config['final_model_dir'], config['final_model_name'])
crop_disease_classes = load_model(final_model_path, model)
model.classifier[1] = nn.Identity()
model.eval()

def predict(path_to_image: str):
    img = Image.open(path_to_image).convert('RGB')
    transform = get_transform('test')
    img = transform(img).unsqueeze(0)
    feature = model(img).detach().numpy()

    # Load the trained SVM model
    svm_model_path = os.path.join(config['final_model_dir'], 'svm_model.joblib')
    svm_model = joblib.load(svm_model_path)

    # Predict the class and confidence
    predict = svm_model.predict_proba(feature)
    predicted_index = predict.argmax()
    confidence = predict[0][predicted_index]
    predicted_crop_disease = crop_disease_classes[predicted_index]
    crop, disease = predicted_crop_disease.split('__', 1)
    return crop, disease, confidence

if __name__ == '__main__':
    predicted_crop, predicted_crop_disease, confidence = predict('./images.jpg')
    print(f"Predicted Crop: {predicted_crop}")
    print(f"Predicted Crop Disease: {predicted_crop_disease}")
    print(f"Confidence: {confidence:.4f}")