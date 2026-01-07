#!/usr/bin/env python3
"""
Extract Layer-Specific PCs Orthogonalized to Global Common PCs.

Algorithm:
1. Load Global Common PCs (already computed)
2. For each layer group (Initial, Middle, Late, Final):
   - Extract GAP features from layers in the group
   - Project out Global Common PC components
   - Run PCA on the residuals
   - Save as layer-specific orthogonalized PCs

This ensures layer-specific PCs capture unique information not present in Global Common PCs.

Output:
- {model}_layer_orth_to_global_pcs.csv: Layer-specific PCs orthogonalized to global common

Usage:
    python extract_layer_pcs_orth_to_global.py \
        --convnext-checkpoint DNNs_model/v9/res_L/convnext_base_regression.pth \
        --image-dir Database \
        --out-dir data_images/dnn_pmods/orthogonalized \
        --n-layer-pcs 10
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.models import convnext_base
from tqdm import tqdm

try:
    import open_clip
except Exception:
    open_clip = None


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_layer_groups(config_path: Path, model_type: str) -> Dict[str, List[str]]:
    """Load layer group configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    if model_type not in config:
        raise ValueError(f"Model type '{model_type}' not found in {config_path}")

    groups_dict = {}
    for group in config[model_type]['groups']:
        label = group['label']
        layers = group['layers']
        groups_dict[label] = layers

    return groups_dict


class ActivationExtractor:
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = set(layer_names)
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.activations.copy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def get_convnext_target_layers(model: nn.Module) -> List[Tuple[str, str]]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU) and name.endswith("block.4"):
            display_name = f"convnext.{name.replace('.', '_')}"
            layers.append((name, display_name))
        elif isinstance(module, nn.Flatten) and name == "classifier.1":
            display_name = f"convnext.{name.replace('.', '_')}"
            layers.append((name, display_name))
    return layers


def get_clip_convnext_target_layers(model: nn.Module) -> List[Tuple[str, str]]:
    layers = []
    trunk = getattr(model, 'trunk', model)

    if hasattr(trunk, 'stages'):
        for stage_idx, stage in enumerate(trunk.stages):
            if hasattr(stage, 'blocks'):
                for block_idx in range(len(stage.blocks)):
                    full_name = f"trunk.stages.{stage_idx}.blocks.{block_idx}.mlp.act"
                    display_name = f"clip.stage{stage_idx}_{block_idx}"
                    layers.append((full_name, display_name))

    if hasattr(model, 'head'):
        layers.append(("head", "clip.head"))

    return layers


def global_average_pooling(activation: torch.Tensor) -> np.ndarray:
    if activation.ndim == 2:
        return activation.cpu().numpy()
    elif activation.ndim == 4:
        pooled = activation.mean(dim=[2, 3])
        return pooled.cpu().numpy()
    elif activation.ndim == 3:
        pooled = activation.mean(dim=2)
        return pooled.cpu().numpy()
    else:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")


def orthogonalize_to_global_pcs(
    features: np.ndarray,
    global_pcs: np.ndarray,
) -> np.ndarray:
    """
    Remove global PC components from features.

    Args:
        features: Shape (n_samples, n_features)
        global_pcs: Shape (n_samples, n_global_pcs) - PC scores for each sample

    Returns:
        Orthogonalized features
    """
    # Normalize global PCs to unit variance for projection
    global_pcs_norm = (global_pcs - global_pcs.mean(axis=0)) / (global_pcs.std(axis=0) + 1e-10)

    # For each feature dimension, regress out the global PCs
    residuals = features.copy()

    for i in range(features.shape[1]):
        y = features[:, i]
        # Least squares: y = X @ beta + residual
        X = global_pcs_norm
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        prediction = X @ beta
        residuals[:, i] = y - prediction

    return residuals


@torch.no_grad()
def extract_layer_pcs_orth_to_global(
    checkpoint_path: Path | None,
    image_paths: List[Path],
    global_common_path: Path,
    layer_groups_config: Path,
    n_layer_pcs: int,
    device: torch.device,
    batch_size: int = 32,
    model_type: str = "convnext",
) -> List[Dict[str, Any]]:
    """Extract layer-specific PCs orthogonalized to global common PCs."""

    # Load global common PCs
    print(f"Loading global common PCs from: {global_common_path}")
    global_df = pd.read_csv(global_common_path)
    print(f"  Loaded {len(global_df)} rows")

    # Load model
    if model_type == "convnext":
        model = convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned_state = {k.replace("model.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned_state, strict=False)
        model.to(device)
        model.eval()

        target_layers = get_convnext_target_layers(model)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif model_type == "clip":
        if open_clip is None:
            raise RuntimeError("open_clip_torch is required")

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "convnext_base",
            pretrained="laion400m_s13b_b51k"
        )
        model = clip_model.visual
        model.to(device)
        model.eval()

        target_layers = get_clip_convnext_target_layers(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    layer_names = [name for name, _ in target_layers]
    name_map = {name: display for name, display in target_layers}

    # Load layer groups
    layer_groups = load_layer_groups(layer_groups_config, model_type)
    print(f"Layer groups: {list(layer_groups.keys())}")

    extractor = ActivationExtractor(model, layer_names)

    # Extract GAP features
    print(f"\nExtracting GAP features from {len(target_layers)} layers...")

    gap_features: Dict[str, List[np.ndarray]] = {}
    image_ids: List[str] = []

    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Extracting features"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))

        batch_images = []
        batch_ids = []

        for i in range(start_idx, end_idx):
            try:
                img = Image.open(image_paths[i]).convert("RGB")
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
                batch_ids.append(image_paths[i].stem)
            except Exception as e:
                print(f"Error loading {image_paths[i]}: {e}")
                continue

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        activations = extractor(batch_tensor)

        for layer_name, activation in activations.items():
            gap = global_average_pooling(activation)
            display_name = name_map.get(layer_name, layer_name)
            if display_name not in gap_features:
                gap_features[display_name] = []
            gap_features[display_name].append(gap)

        image_ids.extend(batch_ids)

    # Concatenate all batches
    for layer_name in gap_features:
        gap_features[layer_name] = np.vstack(gap_features[layer_name])

    extractor.remove_hooks()

    # Get global PC scores aligned with image_ids
    print("\nAligning global PC scores with images...")
    global_df_indexed = global_df.set_index('image_id')

    # Get PC columns
    pc_cols = [c for c in global_df.columns if c.startswith('pc')]
    n_global_pcs = len(pc_cols)
    print(f"  Found {n_global_pcs} global PCs")

    global_scores = np.zeros((len(image_ids), n_global_pcs))
    for i, img_id in enumerate(image_ids):
        if img_id in global_df_indexed.index:
            for j, pc_col in enumerate(pc_cols):
                global_scores[i, j] = global_df_indexed.loc[img_id, pc_col]

    # Process each layer group
    print(f"\nOrthogonalizing layer groups to global common PCs...")
    results = []

    for group_label, group_layer_names in layer_groups.items():
        print(f"\n  Processing group: {group_label}")

        # Collect features for this group
        group_features_list = []
        valid_layers = []

        for layer_display in group_layer_names:
            if layer_display in gap_features:
                group_features_list.append(gap_features[layer_display])
                valid_layers.append(layer_display)

        if not valid_layers:
            print(f"    Warning: No valid layers found for group {group_label}")
            continue

        # Concatenate features for this group
        group_features = np.hstack(group_features_list)
        print(f"    Group features shape: {group_features.shape}")

        # Orthogonalize to global PCs
        orth_features = orthogonalize_to_global_pcs(group_features, global_scores)
        print(f"    After orthogonalization: {orth_features.shape}")

        # Run PCA on orthogonalized features
        n_components = min(n_layer_pcs, orth_features.shape[0], orth_features.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(orth_features)

        layer_scores = pca.transform(orth_features)

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"    PCA: {n_components} components explain {explained_var:.2f}% variance")

        # Save results
        for img_idx, image_id in enumerate(image_ids):
            pc_scores = np.zeros(n_layer_pcs, dtype=np.float32)
            pc_scores[:layer_scores.shape[1]] = layer_scores[img_idx].astype(np.float32)

            results.append({
                "image_id": image_id,
                "module_name": f"{model_type}.{group_label}",
                "pc_scores": pc_scores,
            })

    return results


def save_results_to_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
    max_components: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_id", "module_name"]
    fieldnames.extend([f"pc{i+1}" for i in range(max_components)])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            row = {
                "image_id": item["image_id"],
                "module_name": item["module_name"],
            }
            for i in range(max_components):
                if i < len(item["pc_scores"]):
                    row[f"pc{i+1}"] = float(item["pc_scores"][i])
                else:
                    row[f"pc{i+1}"] = 0.0
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract layer PCs orthogonalized to global common PCs"
    )
    ap.add_argument("--convnext-checkpoint", type=Path, default=None)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data_images/dnn_pmods/orthogonalized"))
    ap.add_argument("--n-layer-pcs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--layer-groups-config", type=Path,
                    default=Path("scripts/dnn_analysis/config/layer_groups.json"))
    ap.add_argument("--disable-convnext", action="store_true")
    ap.add_argument("--disable-clip", action="store_true")
    ap.add_argument("--image-list", type=Path, default=None)

    args = ap.parse_args()

    run_convnext = not args.disable_convnext
    run_clip = not args.disable_clip

    if not run_convnext and not run_clip:
        raise ValueError("All models are disabled")

    if run_convnext and args.convnext_checkpoint is None:
        raise ValueError("--convnext-checkpoint required for ConvNeXt")

    # Collect image paths
    image_paths = sorted(args.image_dir.glob("*.jpg")) + sorted(args.image_dir.glob("*.png"))

    if args.image_list:
        with open(args.image_list, 'r') as f:
            allowed_ids = set(line.strip() for line in f if line.strip())
        normalized_allowed_ids = set(id.zfill(4) for id in allowed_ids)
        image_paths = [
            p for p in image_paths
            if p.stem in normalized_allowed_ids or p.stem in allowed_ids
        ]

    print(f"Processing {len(image_paths)} images")

    device = get_device()
    print(f"Using device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if run_convnext:
        print("\n" + "=" * 60)
        print("EXTRACTING CONVNEXT LAYER PCS (ORTH TO GLOBAL)")
        print("=" * 60)

        global_path = args.out_dir / "convnext_global_common_pcs.csv"
        if not global_path.exists():
            raise ValueError(f"Global common PCs not found: {global_path}")

        results = extract_layer_pcs_orth_to_global(
            args.convnext_checkpoint,
            image_paths,
            global_path,
            args.layer_groups_config,
            args.n_layer_pcs,
            device,
            args.batch_size,
            model_type="convnext",
        )

        output_path = args.out_dir / "convnext_layer_orth_to_global_pcs.csv"
        save_results_to_csv(results, output_path, args.n_layer_pcs)
        print(f"\nSaved: {output_path}")

    if run_clip:
        print("\n" + "=" * 60)
        print("EXTRACTING CLIP LAYER PCS (ORTH TO GLOBAL)")
        print("=" * 60)

        global_path = args.out_dir / "clip_global_common_pcs.csv"
        if not global_path.exists():
            raise ValueError(f"Global common PCs not found: {global_path}")

        results = extract_layer_pcs_orth_to_global(
            None,
            image_paths,
            global_path,
            args.layer_groups_config,
            args.n_layer_pcs,
            device,
            args.batch_size,
            model_type="clip",
        )

        output_path = args.out_dir / "clip_layer_orth_to_global_pcs.csv"
        save_results_to_csv(results, output_path, args.n_layer_pcs)
        print(f"\nSaved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
