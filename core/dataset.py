import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from core.utils import get_transform, get_config, get_target_transform
from torch.utils.data import Dataset, DataLoader
from typing import Literal

class CropDiseaseDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row['image']
        crop_disease = row['crop_disease']

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            crop_disease = self.target_transform(crop_disease)
        
        return image, crop_disease

def load_dataset(config_path):
    images = []
    crop_diseases = []

    config = get_config(config_path)
    data_dir = config['dataset_dir']
    p = Path(data_dir)
    loop = tqdm(list(p.rglob('*')), desc='Loading dataset', leave=False)
    for file in loop:
        if file.is_file():
            img = Image.open(file).convert('RGB')
            images.append(img)
            crop_diseases.append(file.parent.name)

    crop_disease_classes = list(sorted(set(crop_diseases)))
    crop_disease_classes_to_idx = {label:idx for idx, label in enumerate(crop_disease_classes)}

    crop_diseases = [crop_disease_classes_to_idx[crop_disease] for crop_disease in crop_diseases]

    df = pd.DataFrame({
        'image': images,
        'crop_disease': crop_diseases
    })

    config = get_config(config_path)
    classes_config = config['classes']
    classes_dir = classes_config['dir']
    classes_file_name = classes_config['file_name']
    classes_path = os.path.join(classes_dir, classes_file_name)
    os.makedirs(classes_dir, exist_ok=True)

    with open(classes_path, 'w') as f:
        for disease in crop_disease_classes:
            f.write(f"{disease}\n")

    return df, crop_disease_classes

def load_classes(config_path):
    config = get_config(config_path)
    classes_config = config['classes']
    classes_dir = classes_config['dir']
    classes_file_name = classes_config['file_name']
    classes_path = os.path.join(classes_dir, classes_file_name)
    
    with open(classes_path, 'r') as f:
        classes = f.read().splitlines()

    return classes

def build_dataloader(df, mode: Literal['train', 'val', 'test'], config_path):
    config = get_config(config_path)
    batch_size = config['batch_size']

    transform = get_transform(mode, config_path)
    target_transform = get_target_transform()
    dataset = CropDiseaseDataset(df, transform=transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'))
    return dataloader