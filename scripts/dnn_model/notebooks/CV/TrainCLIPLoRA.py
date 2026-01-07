#!/usr/bin/env python3
"""
CLIPのLoRAファインチューニング
"""

import os
import sys
from copy import deepcopy

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

import open_clip
from peft import LoraConfig, get_peft_model

# パス設定
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_PATH)

from src.const import ROOT_PATH
from src.dataset import get_mean_std

print("torch cuda available:", torch.cuda.is_available())

# CLIP設定
CLIP_MODEL_NAME = "convnext_base_w_320"
CLIP_PRETRAINED = "laion_aesthetic_s13b_b82k"


class CLIPRegressionModel(nn.Module):
    """CLIP画像エンコーダ + 回帰ヘッド"""
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        # 出力次元を取得
        with torch.no_grad():
            dummy = torch.randn(1, 3, 320, 320)
            out = self.visual(dummy)
            self.feature_dim = out.shape[-1]

        # 回帰ヘッド
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 4, 1)
        )

    def forward(self, x):
        features = self.visual(x)
        return self.regressor(features)


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

        label = self.label_series[idx + 1]
        return image, label


def get_clip_transforms(mean, std):
    """CLIP用のトランスフォーム"""
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, val_transform


def train_fold(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, fold_idx):
    """1フォールドのトレーニング"""
    scaler = torch.amp.GradScaler()
    best_val_corr = -1
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            all_preds = []
            all_labels = []
            val_loss = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.unsqueeze(1).to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    all_preds.append(outputs.cpu())
                    all_labels.append(labels.cpu())

            preds = torch.cat(all_preds, dim=0)
            labels = torch.cat(all_labels, dim=0)
            val_corr = torch.corrcoef(torch.cat([labels, preds], dim=1).T)[0, 1].item()

            if val_corr > best_val_corr:
                best_val_corr = val_corr
                best_model_state = deepcopy(model.state_dict())

            print(f"  Fold {fold_idx} Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} - Val Corr: {val_corr:.4f}")

    return best_model_state, best_val_corr, preds, labels


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データ読み込み
    resp = pd.read_csv(os.path.join(ROOT_PATH, "data/data_responses_NCNP_2types.csv"))

    # 外れ値除外
    outlier = resp["sub_ID"].unique()[
        (resp.groupby("sub_ID")["res_L"].value_counts().unstack() > 896 * 0.75).any(axis=1)
        | (
            (resp.groupby("sub_ID")["res_L"].unique().apply(lambda x: len(x)) <= 4)
            & (resp.groupby("sub_ID")["res_L"].value_counts().unstack() > 896 * 0.65).any(axis=1)
        )
    ]
    print(f"被験者 {len(outlier)} 人を除外")

    res_mean = (
        resp[~resp["sub_ID"].isin(outlier)]
        .groupby("img")[["res_L", "res_H", "res_T"]]
        .mean()
        .copy()
    )

    # ハイパーパラメータ
    h = 6  # フォールド数
    num_epochs = 50  # CLIPは収束が早いので少なめ
    lr = 1e-5  # 学習率を下げる
    batch_size = 16  # バッチサイズも小さく

    # バージョン管理
    max_v = 1
    for dir in os.listdir(os.path.join(ROOT_PATH, "log")):
        if "v" in dir:
            try:
                v = int(dir.split("v")[-1])
                if v > max_v:
                    max_v = v
            except:
                pass
    version = f"{max_v + 1}"
    print(f"Version: v{version}")

    # 画像ディレクトリ
    image_dir = os.path.join(ROOT_PATH, "Database")

    # CLIP画像の正規化値
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    for col in ["res_L"]:
        print(f"\n{'='*60}")
        print(f"Target: {col}")
        print(f"{'='*60}")

        res = res_mean[col]

        # データセット作成
        _, val_transform = get_clip_transforms(clip_mean, clip_std)
        dataset = ImageDataset(image_dir=image_dir, label_series=res, transform=val_transform)

        # ラベルのビニング（Stratified KFold用）
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
        bins = np.linspace(1, 8, 9)
        binned_labels = np.digitize(labels, bins) - 1

        # K-Fold
        skf = StratifiedKFold(n_splits=h, shuffle=True, random_state=42)

        all_preds = []
        all_labels = []
        fold_corrs = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(binned_labels)), binned_labels), 1):
            print(f"\n--- Fold {fold_idx}/{h} ---")

            # CLIPモデル作成
            clip_model, _, _ = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED
            )

            # 回帰モデル作成
            model = CLIPRegressionModel(clip_model)

            # LoRA設定
            # ConvNeXtのLinear層（MLP）にLoRAを適用
            # modules_to_saveで回帰ヘッドも学習対象に含める
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["fc1", "fc2"],  # ConvNeXtのMLP層
                modules_to_save=["regressor"],  # 回帰ヘッドも学習
                lora_dropout=0.1,
                bias="none",
            )

            # LoRA適用
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            model = model.to(device)

            # データローダー
            train_transform, val_transform = get_clip_transforms(clip_mean, clip_std)

            train_subset = Subset(deepcopy(dataset), train_idx)
            val_subset = Subset(dataset, val_idx)

            train_subset.dataset.transform = train_transform
            val_subset.dataset.transform = val_transform

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

            # 最適化
            criterion = nn.HuberLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

            # トレーニング
            best_state, best_corr, preds, labels_tensor = train_fold(
                model, train_loader, val_loader, criterion, optimizer, num_epochs, device, fold_idx
            )

            all_preds.append(preds)
            all_labels.append(labels_tensor)
            fold_corrs.append(best_corr)

            # メモリ解放
            del model, clip_model
            torch.cuda.empty_cache()

        # 全体の相関
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        overall_corr = torch.corrcoef(torch.cat([all_labels, all_preds], dim=1).T)[0, 1].item()

        print(f"\n{'='*60}")
        print(f"Results for {col}:")
        print(f"  Fold correlations: {[f'{c:.4f}' for c in fold_corrs]}")
        print(f"  Mean fold correlation: {np.mean(fold_corrs):.4f}")
        print(f"  Overall correlation: {overall_corr:.4f}")
        print(f"{'='*60}")

        # 結果保存
        save_dir = os.path.join(ROOT_PATH, "model", f"v{version}", col)
        os.makedirs(save_dir, exist_ok=True)

        results = {
            "avg_val_corr": {"clip_lora": overall_corr},
            "fold_corrs": fold_corrs,
            "actual_predicts": torch.stack([all_labels, all_preds], dim=1),
        }

        dill.dump(results, open(os.path.join(save_dir, "clip_lora_results.pkl"), "wb"))
        print(f"Results saved to {save_dir}/clip_lora_results.pkl")
