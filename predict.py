import torch
import os
from PIL import Image
from model import build_model, load_model
from utils import get_transform, get_config

config = get_config()
model_name = config['model_name']
model = build_model(num_classes=config['num_classes'])
final_model_path = os.path.join(config['final_model_dir'], config['final_model_name'])
crop_disease_classes = load_model(final_model_path, model)
model.eval()

def predict(path_to_image: str):
    img = Image.open(path_to_image).convert('RGB')
    transform = get_transform('test')
    img = transform(img).unsqueeze(0)
    output = model(img)
    
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, predicted_class = torch.max(probs, dim=1)

    predicted_class = crop_disease_classes[predicted_class.item()]
    predicted_crop, predicted_crop_disease = predicted_class.split('__', 1)

    return predicted_crop, predicted_crop_disease, conf.item()

if __name__ == '__main__':
    predicted_crop, predicted_crop_disease, confidence = predict('./images.jpg')
    print(f"Predicted Crop: {predicted_crop}")
    print(f"Predicted Crop Disease: {predicted_crop_disease}")
    print(f"Confidence: {confidence:.4f}")