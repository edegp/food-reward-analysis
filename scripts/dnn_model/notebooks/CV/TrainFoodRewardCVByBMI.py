import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.model import (
    init_convnext_base,
    # init_convnext_tiny,
    init_resnet152,
    # init_resnet50,
    init_resnet152,
    init_vgg16,
    # init_vgg16_freeze,
)
from src.const import ROOT_PATH
from src.train import train

print("torch cuda available: ", torch.cuda.is_available())


class ImageDataset(Dataset):
    def __init__(
        self, image_dir: str, label_series: pd.Series, transform=None, is_train=True
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        )
        self.label_series = label_series.astype("float32")
        self.subject_idxs = self.label_series.index.get_level_values(0).unique()
        self.is_train = is_train

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
            if self.is_train:
                random_idxs = self.subject_idxs[
                    torch.randint(0, len(self.subject_idxs), (5,))
                ]
                label = self.label_series.loc[(random_idxs, idx + 1)].mean()
            else:
                label = self.label_series.loc[(slice(None), idx + 1)].mean()
        return image, label


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(torch.cuda.is_available())

    resp = pd.read_csv(os.path.join(ROOT_PATH, "data/data_responses_NCNP_2types.csv"))
    resp["is_obesity"] = resp["BMI"] >= 25

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
    res_df = resp.loc[
        ~resp["sub_ID"].isin(outlier),
        ["sub_ID", "img", "res_L", "res_T", "res_H", "is_obesity"],
    ].copy()
    resp_without_outlier = resp[~resp["sub_ID"].isin(outlier)]
    print("被験者数", resp_without_outlier["sub_ID"].nunique())
    print(
        "肥満被験者数",
        resp_without_outlier[resp_without_outlier["is_obesity"]]["sub_ID"].nunique(),
    )
    print(
        "非肥満被験者数",
        resp_without_outlier[~resp_without_outlier["is_obesity"]]["sub_ID"].nunique(),
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
    for is_obesity, res in res_df.groupby("is_obesity"):
        for col in res.drop(["is_obesity"], axis=1).columns:
            if col != "res_L":
                continue
            if is_obesity:
                init_callbacks = [
                    init_convnext_base,
                    init_resnet152,
                    init_vgg16,
                ]
            else:
                init_callbacks = [
                    # init_convnext_base,
                    init_resnet152,
                    init_vgg16,
                ]
            data = res.groupby(["sub_ID", "img"])[col].mean()
            dataset = ImageDataset(
                image_dir=image_dir, label_series=data, transform=None
            )
            loss = train(
                dataset,
                init_callbacks,
                version=os.path.join(
                    str(max_v + 1), col, "obesity" if is_obesity else "normal"
                ),
                h=h,
                num_epochs=num_epochs,
                lr=lr,
                batch_size=batch_size,
            )
