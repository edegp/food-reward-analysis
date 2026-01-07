import gc
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr

from copy import deepcopy
import multiprocessing
import pickle
import seaborn as sns

from src.const import index_name, DATA_PATH


def pearson_correlation(y_true, y_pred, average="micro"):
    # y_true and y_pred are 2D arrays with shape (n_samples, n_outputs)
    # 1D arrays are reshaped to 2D arrays
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    n_outputs = y_true.shape[1]
    correlations = []
    for i in range(n_outputs):
        corr_coef = pearsonr(y_true[:, i], y_pred[:, i])[0]
        correlations.append(corr_coef)
    if average == "macro":
        return np.mean(correlations)
    else:
        return np.array(correlations)


def init_ridge_analysis():
    pipe_binary = make_pipeline(
        # StandardScaler(),
        # PCA(n_components=0.8, random_state=42),
        RidgeClassifier(random_state=42),
    )

    f1_scorer = make_scorer(f1_score, average="macro")
    roc_auc_scorer = make_scorer(roc_auc_score, average="macro", multi_class="ovo")
    accuracy_scorer = make_scorer(accuracy_score)
    pipe_regression = make_pipeline(
        # StandardScaler(), PCA(n_components=0.8, random_state=42),
        Ridge(random_state=42)
    )
    multiprocessing.set_start_method("forkserver", force=True)
    pearson_scorer = make_scorer(
        pearson_correlation, average="macro", greater_is_better=True
    )
    search_binary = RandomizedSearchCV(
        pipe_binary,
        {
            "ridgeclassifier__alpha": [0.1, 1, 10, 100, 5000, 100000],
        },
        n_iter=6,
        cv=8,
        n_jobs=-1,
        verbose=0,
        refit="roc_auc",
        scoring={
            "roc_auc": roc_auc_scorer,
            "accuracy": accuracy_scorer,
            "f1": f1_scorer,
        },
    )
    search_regression = RandomizedSearchCV(
        pipe_regression,
        {
            "ridge__alpha": [0.1, 1, 10, 100, 5000, 100000],
        },
        n_iter=6,
        cv=8,
        n_jobs=-1,
        verbose=0,
        scoring=pearson_scorer,
    )
    return (
        pipe_binary,
        pipe_regression,
        # f1_scorer,
        # roc_auc_scorer,
        # accuracy_scorer,
        # pearson_scorer,
        search_binary,
        search_regression,
    )


def get_features_pc(activation_features, save_dir, i):
    save_path = os.path.join(
        save_dir,
        f"pca_{i}.pkl",
    )
    if not os.path.exists(save_path):
        pca = make_pipeline(StandardScaler(), PCA(n_components=0.8, random_state=42))
        activation_features_pc = pca.fit_transform(activation_features)
        with open(
            save_path,
            "wb",
        ) as f:
            pickle.dump(pca, f)
    else:
        with open(
            save_path,
            "rb",
        ) as f:
            pca = pickle.load(f)
            activation_features_pc = pca.transform(activation_features)

    return activation_features_pc


def search_best_binary(
    search_binary,
    X_train,
    Y_train,
    save_dir,
    i,
):
    save_path = os.path.join(
        save_dir,
        f"best_pipe_binary_{i}.pkl",
    )
    if not os.path.exists(save_path):
        best_pipe_binary = deepcopy(search_binary).fit(X_train, Y_train).best_estimator_
        with open(
            save_path,
            "wb",
        ) as f:
            pickle.dump(best_pipe_binary, f)
    else:
        with open(
            save_path,
            "rb",
        ) as f:
            best_pipe_binary = pickle.load(f)

    return best_pipe_binary


def search_best_regression(
    search_regression,
    X_train,
    Y_train,
    save_dir,
    name,
    i,
) -> Pipeline:
    save_path = os.path.join(
        save_dir,
        f"best_pipe_regression_{name}{i}.pkl",
    )
    if not os.path.exists(save_path):
        best_pipe_regression = (
            deepcopy(search_regression).fit(X_train, Y_train).best_estimator_
        )
        with open(
            save_path,
            "wb",
        ) as f:
            pickle.dump(best_pipe_regression, f)
    else:
        with open(
            save_path,
            "rb",
        ) as f:
            best_pipe_regression = pickle.load(f)
    return best_pipe_regression


def run_ridge_analysis(
    result_df,
    pipe_binary: Pipeline,
    pipe_regression: Pipeline,
    search_binary,
    search_regression,
    labels,
    gram_value_is_not_nan,
    save_dir,
    data_dir,
    data_len,
    is_pca: bool = True,
):
    for i in tqdm(range(data_len)):
        print(f"Layer {i}")
        print("pca start")
        data_path = os.path.join(
            data_dir,
            f"intermediate_outputs_by_layer_{i + 1}.pth",
        )
        data = torch.load(
            data_path,
            map_location="cpu",
            weights_only=True,
        )
        activation_features = data.flatten(1).numpy()
        if is_pca:
            activation_features = get_features_pc(activation_features, save_dir, i)
        label_len = len(labels)
        print("labels shape", labels.shape)
        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            inn_train,
            inn_test,
        ) = train_test_split(
            activation_features,
            labels.T,
            gram_value_is_not_nan,
            test_size=0.2,
            shuffle=True,
            random_state=i,
        )

        activation_features = None
        for j in range(label_len):
            label = labels[j]
            Y_train_tmp = Y_train[:, j]
            Y_test_tmp = Y_test[:, j]
            print(f"Label {j} shape: {label.shape}")
            if j >= label_len - 2:
                print("binary start")
                best_pipe_binary = search_best_binary(
                    search_binary, X_train, Y_train_tmp, save_dir, i
                )
                best_pipe_binary = pipe_binary.set_params(
                    ridgeclassifier__alpha=best_pipe_binary.get_params()[
                        "ridgeclassifier__alpha"
                    ]
                ).fit(X_train, Y_train_tmp)
                auc = best_pipe_binary.score(X_test, Y_test_tmp)
                binary_pred = best_pipe_binary.predict(X_test)
                print(auc)
                result_df.iloc[j, i] = roc_auc_score(
                    Y_test_tmp, binary_pred, average=None, multi_class="ovo"
                )
                best_pipe_binary = None
                binary_pred = None
                gc.collect()
            else:
                name = index_name[j]
                print(name)
                if name in ["grams_total", "protein_100g", "fat_100g", "carbs_100g"]:
                    X_train_tmp, X_test_tmp, Y_train_tmp_, Y_test_tmp_ = (
                        X_train[inn_train, :],
                        X_test[inn_test, :],
                        Y_train_tmp[inn_train],
                        Y_test_tmp[inn_test],
                    )
                else:
                    X_train_tmp, X_test_tmp, Y_train_tmp_, Y_test_tmp_ = (
                        X_train,
                        X_test,
                        Y_train_tmp,
                        Y_test_tmp,
                    )
                best_pipe_regression = search_best_regression(
                    search_regression, X_train_tmp, Y_train_tmp_, save_dir, name, i
                )

                best_pipe_regression = pipe_regression.set_params(
                    ridge__alpha=best_pipe_regression.get_params()["ridge__alpha"]
                ).fit(X_train_tmp, Y_train_tmp_)
                corr = best_pipe_regression.score(X_test_tmp, Y_test_tmp_)
                regression_pred = best_pipe_regression.predict(X_test_tmp)
                print(corr)
                result_df.iloc[j, i] = pearson_correlation(
                    Y_test_tmp_, regression_pred, average="micro"
                )
                best_pipe_regression = None
                regression_pred = None
                gc.collect()

    return result_df


def create_label_similarity_matrix(label_values):
    labels_tensor = torch.tensor(label_values)
    num_labels = labels_tensor.size(0)
    similarity_matrix = torch.zeros((num_labels, num_labels))
    # コサイン類似度行列を計算
    for i in range(num_labels):
        for j in range(i + 1):  # iまでループすることで下三角のみを計算
            similarity_matrix[i, j] = torch.abs(
                labels_tensor[i].unsqueeze(0) - labels_tensor[j].unsqueeze(0)
            )
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix


def calc_matrix_corr(matrixs, similarity_matrix):
    label_vector = similarity_matrix.flatten()
    df_corr = pd.DataFrame()
    for i, mat in enumerate(matrixs):
        layer_vector = mat.flatten()
        # ベクトルの平均を計算
        mean_label = torch.mean(label_vector)
        mean_layer = torch.mean(layer_vector)

        # 偏差を計算
        label_diff = label_vector - mean_label
        layer_diff = layer_vector - mean_layer

        # ピアソン相関係数の計算
        numerator = torch.sum(label_diff.cpu() * layer_diff.cpu())
        denominator = torch.sqrt(torch.sum(label_diff**2) * torch.sum(layer_diff**2))
        pearson_correlation = numerator / denominator
        df_corr.loc[i, "layer"] = f"layer {i + 1}"
        df_corr.loc[i, "pearson_correlation"] = pearson_correlation.item()
    df_corr["pearson_correlation"] = df_corr["pearson_correlation"].apply(
        lambda x: round(x, 3)
    )
    return df_corr


def plot_corr_lineplot(ds_corr: pd.Series, VERSION: str):
    fig = plt.figure(figsize=(12, 5))
    sns.lineplot(ds_corr.values)
    plt.xticks(np.arange(min(ds_corr.index), max(ds_corr.index), 1.0))
    plt.show()
    fig.savefig(
        os.path.join(
            DATA_PATH,
            "output",
            "RSA",
            VERSION,
            f"{ds_corr.name}_correlation_lineplot.png",
        )
    )
