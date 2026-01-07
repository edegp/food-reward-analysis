from copy import deepcopy
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset
from torchvision import transforms

from src.dataset import get_mean_std, get_train_transforms, get_val_transforms
from src.model import (
    init_convnext_base,
    init_convnext_tiny,
    init_resnet152,
    init_resnet50,
    init_resnet152,
    init_vgg16,
    init_vgg16_freeze,
)
from src.const import ROOT_PATH
from src.train import train, validation_model

print("torch cuda available: ", torch.cuda.is_available())


class PermutationImageDataset(Dataset):
    """パーミュテーションテスト用データセット

    初期化時にラベルを1回シャッフルし、その対応関係を固定する
    """
    def __init__(self, image_dir: str, label_series: pd.Series, transform=None, seed=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        )
        self.label_series = label_series.astype("float32")

        # 初期化時に1回だけラベルをシャッフル
        if seed is not None:
            np.random.seed(seed)
        self.shuffled_labels = self.label_series.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # シャッフル済みのラベルを使用
        label = self.shuffled_labels[idx]

        return image, label


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(torch.cuda.is_available())

    resp = pd.read_csv(os.path.join(ROOT_PATH, "data/data_responses_NCNP_2types.csv"))

    outlier = resp["sub_ID"].unique()[
        (resp.groupby("sub_ID")["res_L"].value_counts().unstack() > 896 * 0.75).any(
            axis=1
        )
        | (
            (resp.groupby("sub_ID")["res_L"].unique().apply(lambda x: len(x)) <= 4)
            & (
                resp.groupby("sub_ID")["res_L"].value_counts().unstack() > 896 * 0.65
            ).any(axis=1)
        )
    ]
    print("被験者", outlier, len(outlier), "人を除外")
    print(
        "75%以上の試行で同じ選択をしている被験者もしくは4種類以下の選択肢しか選択していないかつ65%で同じ選択の被験者を除外"
    )
    res_mean = (
        resp[~resp["sub_ID"].isin(outlier)]
        .groupby("img")[["res_L", "res_H", "res_T"]]
        .mean()
        .copy()
    )
    resp = None
    init_callbacks = [
        init_convnext_base,
        init_resnet152,
        init_vgg16,
    ]
    max_v = 1

    for dir in os.listdir(os.path.join(ROOT_PATH, "log")):
        if "v" in dir:
            v = int(dir.split("v")[-1])
            if v > max_v:
                max_v = v

    print(f"log/v{max_v + 1}")
    # res_L_mean = resp[~resp["sub_ID"].isin(outlier)].groupby("img")["res_L"].mean()
    # Load dataset
    image_dir = os.path.join(ROOT_PATH, "Database")
    batch_size = 373
    # num_epochs = 250
    num_epochs = 250
    lr = 0.0001
    augment_num = 6
    h = 6  # You can set it to any value you like
    criterion = torch.nn.HuberLoss()
    optimizer_class = torch.optim.AdamW

    for col in res_mean.columns:
        if col != "res_L":
            continue
        print(col)

        res = res_mean[col]
        for init_model_callback in init_callbacks:
            model = init_model_callback()

            for i in range(100):
                # 各イテレーションで異なるシードでラベルをシャッフル
                dataset = PermutationImageDataset(
                    image_dir=image_dir, label_series=res, transform=None, seed=i
                )
                model_copy = deepcopy(model)
                optimizer = optimizer_class(
                    filter(lambda p: p.requires_grad, model_copy.parameters()), lr=lr
                )
                train_dataset, val_dataset = random_split(
                    dataset, [len(dataset) - len(dataset) // 6, len(dataset) // 6]
                )
                mean, std = get_mean_std(train_dataset)

                train_dataset.dataset.transform = get_train_transforms(mean, std)

                val_dataset.dataset.transform = get_val_transforms(mean, std)

                train_datasets = [deepcopy(train_dataset)] * (augment_num - 1)
                train_dataset.dataset.transform = val_dataset.dataset.transform
                train_datasets.append(train_dataset)

                train_loader = DataLoader(
                    dataset=ConcatDataset(train_datasets),
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                )
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=8,
                )
                validation_model(
                    i,
                    model_copy,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    num_epochs,
                    version=os.path.join(f"{max_v + 1}", res.name, str(i)),
                )
