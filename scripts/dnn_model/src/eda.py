import os
from copy import deepcopy
import dill
import clip
from tqdm import tqdm
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from scipy.stats import t
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.stats import ttest_ind

from src import ROOT_PATH, DATA_PATH
from src.dataset import get_mean_std
from src.model import convnext_base, convnext_tiny


warnings.filterwarnings("ignore")


def calc_pvalue(tensor, null_dist=None):
    if isinstance(tensor, torch.Tensor):
        x, y = tensor
        # Compute correlation
        correlation = torch.corrcoef(tensor)[0, 1].cpu().item()
        # Compute t-statistic
        n = x.numel()
    else:
        correlation = tensor

    # permutation test
    if null_dist is not None:
        n_perm = null_dist.shape[1]
        p_value = (np.abs(null_dist.mean(axis=0)) > np.abs(correlation)).sum() / n_perm
        print(f"Permutation test: p-value = {p_value}")
        return correlation, p_value
    if "n" in locals():
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))

        # Compute p-value using SciPy
        df = n - 2
        p_value = t.sf(t_stat, df)  # two-tailed test

        return correlation, p_value
    else:
        return correlation


intermediate_outputs = []


def save_intermediate_activations(
    model: nn.Module,
    data_loader,
    save_dir: str,
    device: torch.device,
):
    """
    model: 任意の nn.Module
    data_loader: (inputs, labels) を吐く DataLoader
    save_dir: 保存先ディレクトリ
    device: torch.device("cuda") など
    """

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # バッファとフック定義
    buffers = []
    intermediate_outputs = []  # フックコールバック用のグローバルバッファ

    def hook_fn(module, inp, out):
        # out は GPU 上の Tensor なので CPU に移動して detach
        intermediate_outputs.append(out.detach().cpu())

    def register_hooks(module: nn.Module):
        """
        モデル内の GELU/ReLU にフックを登録し、buffers に空リストを対応付ける
        """
        for name, child in module.named_children():
            # 必要に応じてスキップ
            if name == "transformer":
                continue

            # 活性化関数層を検出したらフック登録
            print(name)
            if isinstance(child, (nn.GELU, nn.ReLU)) and hasattr(
                child, "register_forward_hook"
            ):
                # buffers に空リストを追加
                buffers.append([])
                child.register_forward_hook(
                    hook_fn
                )  # コールバックは intermediate_outputs に append
                print(f"Registered hook on: {name}")

            # 再帰的に子モジュールも探索
            register_hooks(child)

    # フックを全モデルに登録
    register_hooks(model)

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(
            tqdm(data_loader, total=len(data_loader))
        ):
            # 1) intermediate_outputs をリセット
            intermediate_outputs.clear()

            intermediate_outputs = []
            if isinstance(model, clip.model.CLIP):
                inputs = inputs.unsqueeze(0).to(device, non_blocking=True)
                _ = model.encode_image(inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
                _ = model(inputs)

            # 3) intermediate_outputs → buffers に移植
            for i, act in enumerate(intermediate_outputs):
                buffers[i].append(act)

    # 4) バッチごとにディスク保存＆クリア
    for layer_i, feat_list in enumerate(buffers):
        if not feat_list:
            continue
        batch_feats = torch.cat(feat_list, dim=0)
        print(batch_feats.shape)
        file_path = os.path.join(
            save_dir, f"intermediate_outputs_by_layer_{layer_i + 1}.pth"
        )
        torch.save(batch_feats, file_path)
        buffers[layer_i].clear()
        del batch_feats

    torch.cuda.empty_cache()

    # 最後にフックを外す
    for hook in list(model._forward_hooks.values()):
        hook.remove()


# フックの定義
def hook_fn(module, input, output):
    intermediate_outputs.append(output.cpu())


# モデルのすべてのレイヤーにフックを登録
def register_hooks(module: nn.Module, layer_num=0):
    """
    活性化パターンを取得するためのフックを登録する
    """
    for name, child in module.named_children():
        if name == "transformer":
            continue
        if (
            isinstance(child, nn.GELU)
            # or isinstance(child, clip.model.QuickGELU)
            or isinstance(child, nn.ReLU)
        ) and hasattr(child, "register_forward_hook"):
            for param in child.named_parameters():
                print(param)
            child.register_forward_hook(hook_fn)
            print(name)
            layer_num += 1
        if isinstance(child, nn.Sequential) or isinstance(child, nn.Module):
            # print(name)
            layer_num = register_hooks(
                child, layer_num
            )  # 子レイヤーに対しても再帰的にフックを登録
    return layer_num


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, label_series: np.array, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        )
        self.label_series = label_series

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
            label = self.label_series[idx]

        return image, label


def init_for_intermediate_outputs(labels, VERSION, type_=""):
    """
    Initialize the model and dataset for extracting intermediate outputs
    """
    model = convnext_tiny() if VERSION == "v3" else convnext_base()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    model.load_state_dict(
        torch.load(
            os.path.join(
                ROOT_PATH,
                "model",
                VERSION,
                "res_L",
                type_,
                "like_convnext_tiny_regression.pth"
                if VERSION == "v3"
                else "convnext_base_regression.pth",
            )
        )
    )
    print(model)
    # フックを登録
    global layer_num
    layer_num = 0
    layer_num = register_hooks(model)

    # Load dataset
    image_dir = os.path.join(ROOT_PATH, "Database")
    dataset = ImageDataset(image_dir=image_dir, label_series=labels, transform=None)
    mean, std = get_mean_std(dataset)
    dataset.transform = transforms.Compose(
        [
            v2.ToImage(),
            v2.Resize((240, 240)),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    # device = "cpu"
    data_loader = DataLoader(
        dataset, batch_size=1, pin_memory=device == "cuda", num_workers=4
    )
    # model = torch.jit.script(model)
    return model, data_loader, device, layer_num


def save_intermediate_outputs(model, data_loader, save_dir, device):
    global intermediate_outputs
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    intermediate_outputs_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, total=len(data_loader)):
            # 中間層の出力を保存するためのリスト
            intermediate_outputs = []
            if isinstance(model, clip.model.CLIP):
                inputs = inputs.unsqueeze(0).to(device, non_blocking=True)
                outputs = model.encode_image(inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
            del inputs, outputs
            for i in range(len(intermediate_outputs)):
                if len(intermediate_outputs_list) <= i:
                    intermediate_outputs_list.append([])
                intermediate_outputs_list[i].append(deepcopy(intermediate_outputs[i]))
                # del intermediate_outputs[i]
            intermediate_outputs.clear()
            # del intermediate_outputs
            torch.cuda.empty_cache()
    for layer_idx, batch_list in enumerate(intermediate_outputs_list):
        tensor_to_save = torch.cat(
            batch_list, dim=0
        )  # batch_list は CPU tensor のみならそのまま
        torch.save(
            tensor_to_save,
            os.path.join(
                save_dir, f"intermediate_outputs_by_layer_{layer_idx + 1}.pth"
            ),
        )
        # メモリをすぐ空けたい場合
        intermediate_outputs_list[layer_idx] = None
        del tensor_to_save
        torch.cuda.empty_cache()


def process_layer(i, save_dir):
    tmp = []
    tmp = torch.load(
        os.path.join(
            save_dir,
            f"intermediate_outputs_by_layer_{i + 1}.pth",
        ),
        map_location="cpu",
        weights_only=True,
    )
    tmp_image_num = tmp.size(0)
    matrix = torch.zeros((tmp_image_num, tmp_image_num)).to("cpu")
    if tmp.size(1) == 1:
        for i in range(tmp_image_num):
            for j in range(i + 1):
                # correlation coefficient
                matrix[i, j] = torch.abs(tmp[i] - tmp[j].flatten()).to("cpu")
                matrix[j, i] = matrix[i, j]
    else:
        matrix = 1 - torch.corrcoef(tmp.flatten(1))
        matrix = matrix.tril(diagonal=-1)
    return matrix


def save_layer_matrixs(save_dir, layer_num, n_data_loader=896):
    matrixs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for matrix in tqdm(
            executor.map(lambda i: process_layer(i, save_dir), range(layer_num)),
            total=layer_num,
            desc="Processing Layers",
        ):
            matrixs.append(matrix)

    matrixs = torch.stack(matrixs)
    torch.save(
        matrixs,
        os.path.join(save_dir, "layer_matrixs.pth"),
    )


def get_predict_metrics(VERSION):
    """
    Return the prediction and actual values for each model
    """
    metrics = {}
    with open(os.path.join(DATA_PATH, "output", "clip_res_L_score.pkl"), "rb") as f:
        clip_res_L_score = dill.load(f)

    if "obesity" in os.listdir(os.path.join(ROOT_PATH, "model", VERSION, "res_L")):
        for type_ in ["obesity", "normal"]:
            metric = dill.load(
                open(
                    os.path.join(
                        ROOT_PATH, "model", VERSION, "res_L", type_, "avg_metric.pkl"
                    ),
                    "rb",
                )
            )

            metric["actual_predicts"]["clip"] = torch.stack(
                [
                    torch.cat(
                        [
                            torch.tensor(array)
                            for array in clip_res_L_score[type_]["y_tests"]
                        ]
                    ),
                    torch.cat(
                        [
                            torch.tensor(array)
                            for array in clip_res_L_score[type_]["y_preds"]
                        ]
                    ),
                ],
            )

            metrics[type_] = metric
    else:
        metric = dill.load(
            open(
                os.path.join(ROOT_PATH, "model", VERSION, "res_L", "avg_metric.pkl"),
                "rb",
            )
        )

        metrics["normal"] = metric
    return metrics


def get_corr_log_df(VERSION, path=""):
    # ログファイルのパス
    log_path = f"../../log/{VERSION}/{path}/**/events.out.tfevents.*"

    # マッチするファイルを取得
    event_files = glob.glob(log_path, recursive=True)

    # 取得したファイルリストを確認
    print(f"Found {len(event_files)} event files in {log_path}")
    # print(event_files)

    # 各イベントファイルを読み取る
    df = pd.DataFrame()
    for file_path in event_files:
        try:
            event_acc = EventAccumulator(file_path)
            event_acc.Reload()  # データをロード
            scalar_tags = event_acc.Tags()["scalars"]
            # print(f"Scalar tags in {file_path}: {scalar_tags}")
            index = slice(-4, -1) if VERSION in ["v6", "v16"] else slice(-3, -1)
            model_fold = "-".join(file_path.split("/")[index])
            for tag in scalar_tags:
                if tag == "val_corr":
                    events = event_acc.Scalars(tag)
                    for e in events:
                        df.loc[e.step, model_fold] = e.value
        except Exception as e:
            print(f"Error: {e}")
            continue
    names = (
        ["is_obesity", "model", "fold"]
        if VERSION in ["v6", "v16"]
        else ["model", "fold"]
    )
    df.columns = pd.MultiIndex.from_tuples(
        df.columns.to_series().str.split("-"), names=names
    )
    df.sort_index(axis=1, inplace=True)
    return df


def calc_corr_pvalue_between_model(metrics):
    for model in ["vgg16", "clip", "resnet152", "convnext_base"]:
        normal_data = (
            metrics["normal"]["actual_predicts"][model][-2:].mean(0)
            if model != "clip"
            else metrics["normal"]["actual_predicts"][model].T
        )
        obesity_data = (
            metrics["obesity"]["actual_predicts"][model][-2:].mean(0)
            if model != "clip"
            else metrics["obesity"]["actual_predicts"][model].T
        )
        normal = [
            torch.corrcoef(d.T)[0][1].item() for d in torch.chunk(normal_data, 6, dim=0)
        ]
        obesity = [
            torch.corrcoef(d.T)[0][1].item()
            for d in torch.chunk(obesity_data, 6, dim=0)
        ]
        stat, p = ttest_ind(normal, obesity, equal_var=False)
        print("t検定統計量:", stat)
        print("p値:", p)


def get_pvalue_thr(p):
    """
    Return the threshold for the p-value
    """
    if p < 0.0025:
        return "0.0025"
    elif p < 0.0125:
        return "0.0125"
    elif p < 0.025:
        return "0.025"
    else:
        return ""


def plot_star(p_value, x, y, h):
    """
    Add significance stars to the plot
    """
    if p_value < 0.005:
        plt.text(x, y + h, "***", ha="center", fontsize=24, fontweight="bold")
    elif p_value < 0.025:
        plt.text(x, y + h, "**", ha="center", fontsize=24, fontweight="bold")
    elif p_value < 0.05:
        plt.text(x, y + h, "*", ha="center", fontsize=24, fontweight="bold")


def mergecells(table, ix0, ix1):
    ix0, ix1 = np.asarray(ix0), np.asarray(ix1)
    d = ix1 - ix0
    if not (0 in d and 1 in np.abs(d)):
        raise ValueError(
            "ix0 and ix1 should be the indices of adjacent cells. ix0: %s, ix1: %s"
            % (ix0, ix1)
        )

    if d[0] == -1:
        edges = ("BRL", "TRL")
    elif d[0] == 1:
        edges = ("TRL", "BRL")
    elif d[1] == -1:
        edges = ("BTR", "BTL")
    else:
        edges = ("BTL", "BTR")

    # hide the merged edges
    for ix, e in zip((ix0, ix1), edges):
        table[ix[0], ix[1]].visible_edges = e

    txts = [table[ix[0], ix[1]].get_text() for ix in (ix0, ix1)]
    tpos = [np.array(t.get_position()) for t in txts]

    # center the text of the 0th cell between the two merged cells
    trans = (tpos[1] - tpos[0]) / 2
    if trans[0] > 0 and txts[0].get_ha() == "right":
        # reduce the transform distance in order to center the text
        trans[0] /= 2
    elif trans[0] < 0 and txts[0].get_ha() == "right":
        # increase the transform distance...
        trans[0] *= 2
    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))

    # hide the text in the 1st cell
    txts[1].set_visible(False)
    # hide the background patch of the 1st cell
    table[ix1[0], ix1[1]].set_facecolor("none")


def plot_corr_by_model(corr_df, VERSION):
    # Seaborn の棒グラフ
    fig = plt.figure(figsize=(16, 9), dpi=300)
    sns.barplot(
        data=corr_df,
        x="model",
        y="corr",
        hue="type" if "type" in corr_df.columns else None,
        # palette="Grays",
        width=0.6,
        errorbar=None,
        palette=["#5E5F5F", "#959595", "#D5D5D5", "#D5D5D5"]
        if VERSION == "v9"
        else None,
    )
    h = 0.1

    # for i, row in corr_df.iterrows():
    #     plot_star(
    #         row["p-value"],
    #         i - 0.16
    #         if "type" in corr_df.columns and row["type"] != "obesity"
    #         else i + 0.16,
    #         row["corr"],
    #         h,
    #     )

    # sns.stripplot(data=last_row, x='model', y=0, dodge=False, size=5, marker='o', linewidth=0, color='black')

    # plt.title('Validation Correlation', fontsize=24)
    if VERSION == "v16":
        plt.xticks([], fontsize=36, fontweight="bold")
    else:
        plt.xticks(fontsize=42, fontweight="bold")
    plt.xlabel("", fontsize=36)
    plt.yticks(fontsize=46, fontweight="bold")
    plt.ylabel("Correlation", fontsize=52, labelpad=20, fontweight="bold")
    plt.ylim(0.0, 1.0)

    # if VERSION == "v16":
    #     # テーブルを追加
    #     table_data = (
    #         corr_df.groupby(["model", "type"])["corr"]
    #         .mean()
    #         .round(3)
    #         .reset_index()
    #         .drop("type", axis=1)
    #     )
    #     table_data["model"][1::2] = ""
    #     table = plt.table(
    #         cellText=table_data.values.T,
    #         rowLabels=["モデル", "予測精度"],
    #         cellLoc="center",
    #         loc="bottom",
    #         bbox=[0, -0.35, 1, 0.3],  # [x, y, width, height]
    #     )
    #     fig.canvas.draw()
    #     # table.set_fontweight("bold")
    #     mergecells(table, (0, 0), (0, 1))
    #     mergecells(table, (0, 2), (0, 3))
    #     mergecells(table, (0, 4), (0, 5))
    #     # mergecells(table, (0, 6), (0, 7))

    #     table.auto_set_font_size(False)
    #     for key, cell in table.get_celld().items():
    #         cell.get_text().set_fontweight("bold")
    #         cell.get_text().set_fontsize(28)

    #     table.set_fontsize(28)
    #     table.scale(1.5, 1.5)
    #     table.set_fontweight("bold")

    #     table.auto_set_font_size(False)
    #     table.set_fontsize(20)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=[
            "健常",
            "肥満",
        ],
        fontsize=27,
        loc="upper right",
        bbox_to_anchor=(1, 1.05),
    )

    # グラフとテーブルの間に余白を追加
    # plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()

    plt.show()
    fig.savefig(
        os.path.join(DATA_PATH, "output", "val_corr", f"val_corr_{VERSION}.png"),
        bbox_inches="tight",
    )


def get_corr_df(corr_df, result, VERSION, type_=None):
    if VERSION in ["v9", "v16"]:
        convnext = "convnext_base"
        resnet = "resnet152"
        vgg = "vgg16"
    else:
        convnext = "ConvNeXt"
        resnet = "ResNet"
        vgg = "VGG"
    # バージョンによって切り取るインデックスを変更
    index = slice(-2, None) if VERSION != "v3" else slice(100, 169)
    corr_convnext = calc_pvalue(result[convnext][index].mean(0).T)
    corr_resnet = calc_pvalue(result[resnet][index].mean(0).T)
    corr_vgg = calc_pvalue(result[vgg][index].mean(0).T)

    if VERSION == "v16":
        corr_clip = calc_pvalue(result["clip"])

    print(
        f"ConvNext: r = {corr_convnext}",
        f"ResNet: r = {corr_resnet}",
        f"VGG: r = {corr_vgg}",
        f"CLIP: r = {corr_clip}" if VERSION == "v16" else "",
    )
    models = ["ConvNext", "ResNet", "VGG"]
    corrs = [
        # corr_clip[0],
        corr_convnext[0],
        corr_resnet[0],
        corr_vgg[0],
    ]
    pvalues = [
        # corr_clip[1],
        corr_convnext[1],
        corr_resnet[1],
        corr_vgg[1],
    ]

    corr_df = pd.concat(
        [
            corr_df,
            pd.DataFrame(
                {
                    "model": models,
                    "corr": corrs,
                    "p-value": pvalues,
                }
                if type_ is None
                else {
                    "type": type_,
                    "model": models,
                    "corr": corrs,
                    "p-value": pvalues,
                }
            ),
        ]
    )
    return corr_df
