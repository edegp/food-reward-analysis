import os
import pandas as pd
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset
from torchvision import transforms

from src.model import (
    init_convnext_base,
    init_convnext_tiny,
    init_resnet152,
    init_resnet50,
    init_resnet152,
    init_vgg16,
    init_vgg16_freeze,
)
from src.train import h_hold_fine_tune_regression
from src.const import ROOT_PATH


print(torch.cuda.is_available())


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, label_series: pd.Series, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = image.mean(axis=(1, 2)).sum()

        return image, label


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Load dataset
    image_dir = os.path.join(ROOT_PATH, "Database")
    dataset = ImageDataset(
        image_dir=image_dir, label_series=pd.Series(), transform=None
    )
    # Define the h value (number of folds)
    h = 6  # You can set it to any value you like

    for init_model_callback in [init_resnet152, init_vgg16]:
        model_name = "_".join(init_model_callback.__name__.split("_")[1:])
        batch_size = 896 // 2
        num_epochs = 40
        # if 'vgg16' in init_model_callback.__name__:
        #     num_epochs = 100
        lr = 0.001
        augment_num = 3
        avg_val_loss = h_hold_fine_tune_regression(
            [init_model_callback],
            dataset,
            nn.HuberLoss(),
            optim.AdamW,
            h=h,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            num_augmentations=augment_num,
            trial_num=1,
        )
