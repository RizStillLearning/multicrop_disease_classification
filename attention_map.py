import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
from pathlib import Path

# Import the GradCAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import get_config, get_transform
from model import build_model, load_model

# 1. Initialize configuration and device
config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Initialize model and load weights
num_classes = config['num_classes']
model = build_model(num_classes=num_classes)

backbone_dir = config['backbone_dir']
backbone_name = config['backbone_name']
backbone_path = os.path.join(backbone_dir, backbone_name)
load_model(backbone_path, model)

model = model.to(device)
model.eval()

# 3. Get 5 random sample images
data_dir = Path(config['dataset_dir'])
all_images = list(data_dir.rglob('*.jpg')) + list(data_dir.rglob('*.png')) + list(data_dir.rglob('*.jpeg'))
unhealthy_images = [img for img in all_images if 'Healthy' not in img.parent.name]
sample_img_paths = random.sample(unhealthy_images, 5)

# 4. Setup Grad-CAM from the library
# The target layer is the last convolutional block in EfficientNet
target_layers = [model.features[-1]]

# Initialize the CAM generator
cam_generator = GradCAM(model=model, target_layers=target_layers)

# 5. Plotting Grid
fig, axes = plt.subplots(3, 5, figsize=(20, 15))

for idx, sample_img_path in enumerate(sample_img_paths):
    true_label = sample_img_path.parent.name
    
    # Load and preprocess the image
    transform = get_transform('val')
    img_pil = Image.open(sample_img_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Generate the raw grayscale CAM
    grayscale_cam = cam_generator(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :] # Extract the first image in the batch

    # Visualization using cv2 and show_cam_on_image
    img_np = np.array(img_pil.resize((config['image_size'], config['image_size'])))
    img_np = img_np.astype(np.float32) / 255.0
    overlayed_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Original Image
    axes[0, idx].imshow(img_np)
    axes[0, idx].set_title(true_label)
    axes[0, idx].axis('off')

    # Raw Attention Map
    axes[1, idx].imshow(grayscale_cam, cmap='jet')
    axes[1, idx].set_title('Attention Map (Grad-CAM)')
    axes[1, idx].axis('off')

    # Overlayed Image
    axes[2, idx].imshow(overlayed_img)
    axes[2, idx].set_title('Overlayed Heatmap')
    axes[2, idx].axis('off')

plot_dir = Path(config['plot_dir'])
plot_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_dir / 'attention_maps.png')

plt.show()