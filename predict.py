import torch
from PIL import Image
from model import build_model, load_model
from utils import get_transform, get_config

config = get_config()
model_name = config['model_name']
model = build_model(model_name)
crop_disease_classes = load_model(model_name, model)

config = get_config()
model_name = config['model_name']
model = build_model(model_name)
crop_disease_classes, disease_to_crop_mapping = load_model(model_name, model)

def predict(path_to_image: str):
    img = Image.open(path_to_image).convert('RGB')
    transform = get_transform('test')
    img = transform(img).unsqueeze(0)
    output = model(img)
    
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, predicted_class = torch.max(probs, dim=1)

    predicted_crop_disease = crop_disease_classes[predicted_class.item()]
    predicted_crop = disease_to_crop_mapping[predicted_crop_disease]

    return (predicted_crop, float("{:.4f}".format(conf.item()))), (predicted_crop_disease, float("{:.4f}".format(conf.item())))

if __name__ == '__main__':
    predicted_crop, predicted_crop_disease = predict('./images.jpeg')
    print(predicted_crop)
    print(predicted_crop_disease)