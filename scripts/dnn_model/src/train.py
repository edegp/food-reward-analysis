from copy import deepcopy
import os
import shutil
from typing import Callable

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torch.optim import Optimizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim as optim

from .const import DATA_PATH, ROOT_PATH
from .dataset import get_mean_std, get_train_transforms, get_val_transforms

# GPU高速化設定
torch.backends.cudnn.benchmark = True  # ConvNet最適化
# PyTorch 2.9+ の新しいTF32 API
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"


def getEnvironment():
    try:
        env = get_ipython().__class__.__name__
        if env == "ZMQInteractiveShell":
            return "Jupyter"
        elif env == "TerminalInteractiveShell":
            return "IPython"
        else:
            return "OtherShell"
    except NameError:
        return "Interpreter"


# デバイス選択関数 (TPU > GPU > CPUの順に優先)
def get_device():
    if torch.cuda.is_available():
        # GPUの設定
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        # CPUの設定
        device = torch.device("cpu")
        device_type = "cpu"
    print(f"Using {device} device")
    return device, device_type


def train(
    dataset,
    init_callbacks,
    version,
    loss=nn.HuberLoss(),
    optimizer=optim.AdamW,
    h=6,
    num_epochs=250,
    lr=0.0001,
    batch_size=32,
    num_augmentations=6,
    trial_num=1,
):
    print(
        "h:",
        h,
        "num_epochs:",
        num_epochs,
        "lr:",
        lr,
        "batch_size:",
        batch_size,
        "num_augmentations:",
        num_augmentations,
    )

    avg_val_loss = h_hold_fine_tune_regression(
        init_callbacks,
        dataset,
        criterion=loss,
        optimizer_class=optimizer,
        h=h,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        num_augmentations=num_augmentations,
        trial_num=trial_num,
        version=version,
    )
    return avg_val_loss


# モデルの学習と検証
def validation_model(
    index,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=10,
    version="",
):
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    device, device_type = get_device()
    env = getEnvironment()
    print(env)
    if env == "Jupyter" or env == "OtherShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    model.to(device)

    # channels_last メモリフォーマット (ConvNet高速化)
    model = model.to(memory_format=torch.channels_last)

    # torch.compileで高速化 (default: CUDA Graphsなしで安定)
    model = torch.compile(model, mode="default")

    writer = SummaryWriter(
        log_dir=os.path.join(
            ROOT_PATH, f"log/v{version}/{model.__class__.__name__}/fold_{index}"
        )
    )

    epoch_pbar = tqdm(range(num_epochs), total=num_epochs, leave=False)

    fold_train_loss = 0.0
    fold_val_loss = 0.0
    val_corr = 0.0

    # Early stopping用 (移動平均で判定)
    best_avg_val_loss = float('inf')
    val_loss_history = []
    patience = 6  # 6回検証改善なしで停止 (=30 epoch) - 長めに学習
    patience_counter = 0

    print(f"Fold {index} Training..., device: {device} {len(train_loader)}")
    all_actual_predicts = []

    # Training phase
    for epoch in epoch_pbar:
        running_loss = torch.tensor(0.0, device=device)  # GPU上で累積
        model.train()
        train_pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.unsqueeze(1).to(device, non_blocking=True),
            )

            optimizer.zero_grad(set_to_none=True)  # Noneに設定で高速化

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.detach()  # GPU上で累積、同期なし

        epoch_train_loss = running_loss.item() / len(train_loader)  # エポック終了時のみ同期
        fold_train_loss += epoch_train_loss

        # Validation phase
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            total_val_loss = torch.tensor(0.0, device=device)  # GPU上で累積
            results_list = []
            all_labels_list = []
            with torch.inference_mode():  # no_gradより高速
                for inputs, labels in val_loader:
                    inputs, labels = (
                        inputs.to(device, non_blocking=True),
                        labels.unsqueeze(1).to(device, non_blocking=True),
                    )

                    with torch.amp.autocast(
                        device_type=device_type, dtype=torch.bfloat16
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    total_val_loss += loss.detach()  # GPU上で累積
                    results_list.append(outputs.cpu())  # CPUに移動
                    all_labels_list.append(labels.cpu())  # CPUに移動

                results = torch.cat(results_list, dim=0)
                all_labels = torch.cat(all_labels_list, dim=0)

            epoch_val_loss = total_val_loss.item() / len(val_loader)  # エポック終了時のみ同期
            fold_val_loss += epoch_val_loss

            actual_predict = torch.cat((all_labels, results), dim=1)  # CPUに保持
            all_actual_predicts.append(actual_predict.cpu())  # CPUに蓄積
            val_corr = torch.corrcoef(actual_predict.T)[0, 1]
            if torch.isnan(val_corr):
                print("val_corr is nan")
                print(actual_predict)
                val_corr = 0.0
            if val_corr > 0.99:
                return (
                    model,
                    fold_train_loss / epoch,
                    fold_val_loss / epoch,
                    torch.stack(all_actual_predicts, dim=0),  # 全エポック保持
                )

            # Early stopping check (移動平均で判定)
            val_loss_history.append(epoch_val_loss)
            if len(val_loss_history) >= 5:
                avg_val_loss = sum(val_loss_history[-5:]) / 5
                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        writer.add_scalar("train_loss", epoch_train_loss, epoch)
        writer.add_scalar("val_loss", epoch_val_loss, epoch)
        writer.add_scalar("val_corr", val_corr, epoch)

        if isinstance(epoch_pbar, tqdm):
            epoch_pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {fold_train_loss / (epoch + 1):.4f} - Val Loss: {fold_val_loss / (epoch + 1):.4f} - Val Corr: {val_corr:.4f}"
            )
            epoch_pbar.refresh()

    if isinstance(epoch_pbar, tqdm):
        epoch_pbar.close()

    writer.close()
    return (
        model,
        fold_train_loss / num_epochs,
        fold_val_loss / num_epochs,
        torch.stack(all_actual_predicts, dim=0),  # 全エポック保持 (epoch, val_len, 2)
    )


# ファインチューニングを行う関数
def h_hold_fine_tune_regression(
    init_model_callbacks: [Callable[(...), nn.Module]],
    dataset: Dataset,
    criterion: nn.Module,
    optimizer_class: Optimizer,
    h=6,
    num_epochs=20,
    lr=0.0001,
    batch_size=32,
    num_augmentations=6,
    trial_num=None,
    seed=42,
    version=1,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device, device_type = get_device()
    env = getEnvironment()
    if env == "Jupyter":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # DataLoaderでpin_memory=Trueを設定
    def create_dataloader(subset):
        return DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
            persistent_workers=True, prefetch_factor=2  # 8*2=16コアに最適
        )

    # KFoldの設定
    skf = StratifiedKFold(n_splits=h, shuffle=True, random_state=trial_num)

    avg_train_loss = {}
    avg_val_loss = {}
    avg_val_corr = {}
    actual_predicts = {}
    best_model = {}
    best_val_loss = {}

    # データセットのラベルを取得
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    n_bins = 8  # ビンの数（例: 4つのカテゴリに分割）
    bins = np.linspace(1, 8, n_bins + 1)  # ビンの境界を定義
    binned_labels = np.digitize(labels, bins) - 1

    for init_model_callback in init_model_callbacks:
        # torchinductorキャッシュをクリア（CUDA Graphsエラー防止）
        cache_dir = f"/tmp/torchinductor_{os.environ.get('USER', 'user')}"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared torchinductor cache: {cache_dir}")

        fold = 1
        pbar = tqdm(
            skf.split(np.zeros(len(binned_labels)), binned_labels), total=h, leave=False
        )
        model = init_model_callback()
        print(model)
        model_name = "_".join(init_model_callback.__name__.split("_")[1:])

        # CV分割インデックスを保存（encoding分析で同じ分割を使うため）
        cv_indices = []

        for train_idx, val_idx in pbar:
            cv_indices.append({'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist()})
            model_copy = deepcopy(model)
            decay_params, no_decay_params = [], []
            for name, param in model_copy.named_parameters():
                if not param.requires_grad:
                    continue
                # If this param belongs to the last layer → decay, otherwise no_decay
                if not name.endswith("bias") and (
                    (model_name == "convnext_base" and name.startswith("classifier.3."))
                    or name.startswith("classifier.7.")
                    or name.startswith("fc.1.")
                ):
                    print(f"Decay: {name}")
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

            optimizer = optimizer_class(
                [
                    {"params": no_decay_params, "weight_decay": 0.01},
                    {
                        "params": decay_params,
                        "weight_decay": 0.1,
                    },
                ],
                lr=lr,
                fused=True,  # CUDA fused実装で高速化
            )
            # Train and validation subsets
            train_subsampler = Subset(deepcopy(dataset), train_idx)
            val_subsampler = Subset(dataset, val_idx)

            mean, std = get_mean_std(train_subsampler)

            train_subsampler.dataset.transform = get_train_transforms(mean, std)

            val_subsampler.dataset.transform = get_val_transforms(mean, std)
            val_subsampler.dataset.is_train = False

            if fold == 1 and model_name == "convnext_base":
                for i, ((image, label), (val_image, val_label)) in enumerate(
                    zip(train_subsampler, val_subsampler)
                ):
                    if i > 30:
                        break
                    fig = plt.figure(figsize=(10, 10))
                    image = image.permute(1, 2, 0) * std + mean
                    plt.imshow(image)
                    plt.title(f"Label: {label}")
                    fig.savefig(
                        os.path.join(
                            DATA_PATH, "image", "train", f"likely_image_{i}.png"
                        ),
                        dpi=300,
                    )
                    plt.close(fig)
                    fig = plt.figure(figsize=(10, 10))
                    plt.imshow(val_image.permute(1, 2, 0) * std + mean)
                    plt.title(f"Label: {val_label}")
                    fig.savefig(
                        os.path.join(
                            DATA_PATH, "image", "val", f"likely_image_{i}.png"
                        ),
                        dpi=300,
                    )
                    plt.close(fig)

            train_datasets = [deepcopy(train_subsampler)] * (num_augmentations - 1)
            # Add the original training set to the list of datasets
            train_subsampler.dataset.transform = val_subsampler.dataset.transform
            # Create data loaders
            train_loader = create_dataloader(
                ConcatDataset(train_datasets + [train_subsampler])
            )
            val_loader = create_dataloader(val_subsampler)

            del train_subsampler
            if not os.path.exists(os.path.join(ROOT_PATH, f"model/v{version}/")):
                os.makedirs(os.path.join(ROOT_PATH, f"model/v{version}/"))

            # if fold == 1 or fold == 2 and model_name == "convnext_base":
            # continue
            if fold == 1:
                actual_predicts[model_name] = []
            trained_model, fold_train_loss, fold_val_loss, all_actual_predict = (
                validation_model(
                    fold,
                    model_copy,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    num_epochs,
                    version=version,
                )
            )
            # Accumulate the results for averaging
            avg_train_loss[model_name] = (
                avg_train_loss.get(model_name, 0) + fold_train_loss
            )
            avg_val_loss[model_name] = avg_val_loss.get(model_name, 0) + fold_val_loss
            actual_predicts[model_name].append(
                all_actual_predict
            )  # all_actual_predict (epoch, val_len, 2)
            # 相関計算は各Foldの最後のエポックを使用
            last_preds = [pred[-1] for pred in actual_predicts[model_name]]  # [(val_len, 2), ...]
            corr = torch.corrcoef(
                torch.cat(last_preds, dim=0).T
            )[0, 1]  # (fold * val_len, 2)
            pbar.set_description(
                f"Fold {fold} Training Loss: {fold_train_loss:.4f} Validation Loss: {fold_val_loss:.4f} Validation Correlation: {corr:.4f}"
            )

            # 各foldのモデルを個別に保存（encoding分析でのリーク防止用）
            fold_model_dir = os.path.join(ROOT_PATH, f"model/v{version}/folds")
            if not os.path.exists(fold_model_dir):
                os.makedirs(fold_model_dir)
            torch.save(
                trained_model.cpu().state_dict(),
                os.path.join(fold_model_dir, f"{model_name}_fold{fold}.pth"),
            )
            trained_model.to("cuda")  # GPUに戻す

            # Save the best model based on validation loss
            if fold_val_loss < best_val_loss.get(model_name, float("inf")):
                best_val_loss[model_name] = fold_val_loss
                best_model[model_name] = trained_model.cpu().state_dict()
                print(
                    f"Fold {fold}: New best model for {model_name} with validation loss {fold_val_loss:.4f} corr {corr:.4f}"
                )
                # モデルをCPUに移動して保存
                torch.save(
                    best_model[model_name],
                    os.path.join(
                        ROOT_PATH, f"model/v{version}/{model_name}_regression.pth"
                    ),
                )
            del trained_model
            torch.cuda.empty_cache()
            fold += 1

        # CV分割インデックスを保存
        import json
        cv_indices_path = os.path.join(ROOT_PATH, f"model/v{version}/folds/cv_indices.json")
        with open(cv_indices_path, 'w') as f:
            json.dump(cv_indices, f)
        print(f"CV indices saved to {cv_indices_path}")

    # Average results across all folds
    for model_name in avg_train_loss.keys():
        avg_train_loss[model_name] /= h
        avg_val_loss[model_name] /= h

        # actual_predicts[model_name]はリストのまま保持 [(epoch, val_len, 2), ...]
        # 相関計算は各Foldの最後のエポックを使用
        last_preds = [pred[-1] for pred in actual_predicts[model_name]]
        avg_val_corr[model_name] = torch.corrcoef(
            torch.cat(last_preds, dim=0).T
        )[0, 1]

        print(f"\nAverage Training Loss: {avg_train_loss[model_name]:.4f}")
        print(f"Average Validation Loss: {avg_val_loss[model_name]:.4f}")
        print(f"Average Validation Correlation: {avg_val_corr[model_name]:.4f}")

    dill.dump(
        {
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "avg_val_corr": avg_val_corr,
            "actual_predicts": actual_predicts,
        },
        open(
            os.path.join(ROOT_PATH, f"model/v{version}/avg_metric.pkl"),
            "wb",
        ),
    )
    return avg_val_loss
