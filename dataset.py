import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from utils import get_transform, get_config, get_target_transform
from torch.utils.data import Dataset, DataLoader
from typing import Literal
from sklearn.model_selection import train_test_split

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
        crop = row['crop']
        crop_disease = row['crop_disease']

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            crop = self.target_transform(crop)
            crop_disease = self.target_transform(crop_disease)
        
        return image, (crop, crop_disease)

def load_dataset(data_dir):
    images = []
    crops = []
    crop_diseases = []

    p = Path(data_dir)
    for file in p.rglob('*'):
        if file.is_file() and file.suffix in ['.jpg', '.jpeg', '.png']:
            img = Image.open(file).convert('RGB')
            images.append(img)
            crops.append(file.parent.parent.name)
            crop_diseases.append(file.parent.name)

    crop_classes = list(sorted(set(crops)))
    crop_disease_classes = list(sorted(set(crop_diseases)))

    crop_classes_to_idx = {label:idx for idx, label in enumerate(crop_classes)}
    crop_disease_classes_to_idx = {label:idx for idx, label in enumerate(crop_disease_classes)}

    crops = [crop_classes_to_idx[crop] for crop in crops]
    crop_diseases = [crop_disease_classes_to_idx[crop_disease] for crop_disease in crop_diseases]

    df = pd.DataFrame({
        'image': images,
        'crop': crops,
        'crop_disease': crop_diseases
    })

    return df, crop_classes, crop_disease_classes

def split_dataset(df, train_size=0.6, val_size=0.2, test_size=0.2):
    train_df, temp_df = train_test_split(df, train_size=train_size)
    val_df, test_df = train_test_split(temp_df, train_size=val_size/(1 - train_size))
    return train_df, val_df, test_df

def build_dataloader(df, mode: Literal['train', 'val', 'test']):
    config = get_config()
    batch_size = config['batch_size']

    transform = get_transform(mode)
    target_transform = get_target_transform()
    dataset = CropDiseaseDataset(df, transform=transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'))
    return dataloader