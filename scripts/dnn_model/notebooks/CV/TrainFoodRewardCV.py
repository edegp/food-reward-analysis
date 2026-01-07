import os
import pandas as pd
import torch
from PIL import Image
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
from src.const import ROOT_PATH
from src.train import train

print("torch cuda available: ", torch.cuda.is_available())


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, label_series: pd.Series, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        )
        self.label_series = label_series.astype("float32")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Replace 0 with actual label if available
        if self.label_series is None:
            label = image.mean()
        else:
            label = self.label_series[idx + 1]

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
        init_resnet152,
        init_vgg16,
    ]
    max_v = 1

    for dir in os.listdir(os.path.join(ROOT_PATH, "log")):
        if dir.startswith("v") and dir[1:].isdigit():
            v = int(dir[1:])
            if v > max_v:
                max_v = v

    print(f"log/v{max_v + 1}")
    # res_L_mean = resp[~resp["sub_ID"].isin(outlier)].groupby("img")["res_L"].mean()
    # Load dataset
    image_dir = os.path.join(ROOT_PATH, "Database")
    batch_size = 392  # 12バッチ (4704 / 392 = 12)
    num_epochs = 250  # 精度重視
    lr = 0.0001  # v9と同じ学習率
    augment_num = 6
    h = 8  # 8-fold CV
    for col in res_mean.columns:
        if col != "res_L":
            continue
        print(col)

        res = res_mean[col]
        dataset = ImageDataset(image_dir=image_dir, label_series=res, transform=None)
        loss = train(
            dataset,
            init_callbacks,
            version=os.path.join(
                f"{max_v + 1}",
                res.name,
            ),
            h=h,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
        )
