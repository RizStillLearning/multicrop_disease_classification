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
sample_img_paths = random.sample(all_images, 5)

# 4. Setup Grad-CAM from the library
# The target layer is the last convolutional block in EfficientNet
target_layers = [model.features[-1]]

# Initialize the CAM generator
cam_generator = GradCAM(model=model, target_layers=target_layers)

# 5. Plotting Grid
fig, axes = plt.subplots(5, 3, figsize=(40, 30))

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
    axes[idx, 0].imshow(img_np)
    axes[idx, 0].set_title(f'Original Image\nTrue Class: {true_label}')
    axes[idx, 0].axis('off')

    # Raw Attention Map
    axes[idx, 1].imshow(grayscale_cam, cmap='jet')
    axes[idx, 1].set_title('Attention Map (Grad-CAM)')
    axes[idx, 1].axis('off')

    # Overlayed Image
    axes[idx, 2].imshow(overlayed_img)
    axes[idx, 2].set_title('Overlayed Heatmap')
    axes[idx, 2].axis('off')

plt.tight_layout(h_pad=10.0)

plot_dir = Path(config['plot_dir'])
plot_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_dir / 'attention_maps.png')

plt.show()