import torch
from PIL import Image
from model import build_model, load_model
from utils import get_transform, get_config

config = get_config()
model_name = config['model_name']
model = build_model(model_name)
crop_classes, crop_disease_classes = load_model(model_name, model)

def predict(path_to_image: str):
    img = Image.open(path_to_image).convert('RGB')
    transform = get_transform('test')
    img = transform(img).unsqueeze(0)
    crop_output, crop_disease_output = model(img)
    
    crop_probs = torch.nn.functional.softmax(crop_output, dim=1)
    crop_conf, predicted_crop_class = torch.max(crop_probs, dim=1)

    crop_disease_probs = torch.nn.functional.softmax(crop_disease_output, dim=1)
    crop_disease_conf, predicted_crop_disease_class = torch.max(crop_disease_probs, dim=1)

    predicted_crop = crop_classes[predicted_crop_class.item()]
    predicted_crop_disease = crop_disease_classes[predicted_crop_disease_class.item()]

    return (predicted_crop, float("{:.4f}".format(crop_conf.item()))), (predicted_crop_disease, float("{:.4f}".format(crop_disease_conf.item())))

if __name__ == '__main__':
    predicted_crop, predicted_crop_disease = predict('./images.jpeg')
    print(predicted_crop)
    print(predicted_crop_disease)