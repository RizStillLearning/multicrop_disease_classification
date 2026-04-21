import yaml
import torch
import os
from typing import Literal
from torchvision import transforms

def get_config(config_file='config.yaml'):
    config = None
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform(mode: Literal['train', 'val', 'test']):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    config = get_config()
    img_size = config['image_size']

    transform_dict = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=mean, std=std),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=mean, std=std),
        ]),
    }

    return transform_dict[mode]

def get_target_transform():
    return transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))

def save_checkpoint(checkpoint_dir, checkpoint_name, model_state, optimizer_state, cur_epoch, best_val_loss):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save({
        'model_state_dict': model_state.state_dict(),
        'optimizer_state_dict': optimizer_state.state_dict(),
        'cur_epoch': cur_epoch,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, model, best_model_state, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    best_model_state.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    cur_epoch = checkpoint['cur_epoch']
    best_val_loss = checkpoint['best_val_loss']
    return cur_epoch, best_val_loss