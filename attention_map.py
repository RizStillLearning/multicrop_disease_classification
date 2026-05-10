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

# ── 1. Config & device ──────────────────────────────────────────────────────
config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = config['num_classes']
k_fold = config['k_fold']

# ── 2. Load ALL backbone folds ───────────────────────────────────────────────
backbone_dir = config['backbone_dir']
backbone_models = []
backbone_names  = []

for fold_idx in range(1, k_fold + 1):
    backbone_name = f'backbone_fold_{fold_idx}.pth'
    backbone_path = os.path.join(backbone_dir, backbone_name)
    if not os.path.exists(backbone_path):
        print(f"[WARN] {backbone_name} not found, skipping.")
        continue

    model = build_model(num_classes=num_classes)
    load_model(backbone_path, model)
    model = model.to(device)
    model.eval()

    backbone_models.append(model)
    backbone_names.append(f'Fold {fold_idx}')
    print(f"Loaded {backbone_name}")

print(f"\nEnsemble size: {len(backbone_models)} models\n")

# ── 3. Build one GradCAM generator per backbone ──────────────────────────────
cam_generators = [
    GradCAM(model=m, target_layers=[m.features[-1]])
    for m in backbone_models
]

# ── 4. Sample 5 random unhealthy images ──────────────────────────────────────
data_dir = Path(config['dataset_dir'])
all_images = (
    list(data_dir.rglob('*.jpg'))
    + list(data_dir.rglob('*.png'))
    + list(data_dir.rglob('*.jpeg'))
)
unhealthy_images = [img for img in all_images if 'Healthy' not in img.parent.name]
sample_img_paths = random.sample(unhealthy_images, 5)

# ── 5. Build the figure ───────────────────────────────────────────────────────
#   Rows: original | ensemble raw CAM | ensemble overlay
#   Cols: one per sample image
n_rows = 3
n_cols = len(sample_img_paths)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
fig.suptitle('Ensemble Grad-CAM Attention Maps', fontsize=16, fontweight='bold', y=1.02)

transform = get_transform('val')
img_size  = config['image_size']

row_labels = ['Original', 'Ensemble CAM', 'Ensemble Overlay']

for col, sample_img_path in enumerate(sample_img_paths):
    true_label = sample_img_path.parent.name

    # Load image
    img_pil = Image.open(sample_img_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Normalised numpy image for overlaying
    img_np = np.array(img_pil.resize((img_size, img_size))).astype(np.float32) / 255.0

    # Generate CAM from every fold and average them
    fold_cams = [
        cam_gen(input_tensor=input_tensor, targets=None)[0]
        for cam_gen in cam_generators
    ]
    ensemble_cam     = np.mean(np.stack(fold_cams, axis=0), axis=0)
    ensemble_overlay = show_cam_on_image(img_np, ensemble_cam, use_rgb=True)

    # ── Row 0: Original image
    axes[0, col].imshow(img_np)
    axes[0, col].set_title(true_label, fontsize=10, fontweight='bold')
    axes[0, col].axis('off')

    # ── Row 1: Raw ensemble CAM
    axes[1, col].imshow(ensemble_cam, cmap='jet', vmin=0, vmax=1)
    axes[1, col].set_title('Raw Ensemble Heatmap', fontsize=9)
    axes[1, col].axis('off')

    # ── Row 2: Ensemble overlay
    axes[2, col].imshow(ensemble_overlay)
    axes[2, col].set_title('Ensemble Overlay', fontsize=9, color='darkgreen', fontweight='bold')
    axes[2, col].axis('off')

# Add row labels on the left
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=15,
                                 va='center', fontweight='bold')

plt.tight_layout()

# ── 6. Save ───────────────────────────────────────────────────────────────────
plot_dir = Path(config['plot_dir'])
plot_dir.mkdir(parents=True, exist_ok=True)
out_path = plot_dir / 'ensemble_attention_maps.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")

plt.show()