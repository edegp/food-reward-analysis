#!/usr/bin/env python3
"""
Extract Global Common PCs across ALL layers.

This creates a single set of common PCs that are shared across all layer groups,
representing the most fundamental shared visual features (color, brightness, etc.).

Algorithm:
1. Load GAP features from all layers (from orthogonalized extraction)
2. Concatenate all layer features horizontally
3. Run PCA on the concatenated features
4. Save the first N components as global common PCs

Output:
- {model}_global_common_pcs.csv: Single set of common PCs for all images

Usage:
    python extract_global_common_pcs.py \
        --convnext-checkpoint DNNs_model/v9/res_L/convnext_base_regression.pth \
        --image-dir Database \
        --out-dir data_images/dnn_pmods/orthogonalized \
        --n-global-pcs 4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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


def get_convnext_target_layers(model: nn.Module) -> List[Tuple[str, str]]:
    """Get target layer names for ConvNeXt."""
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
    """Get target layer names for CLIP ConvNeXt."""
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


@torch.no_grad()
def extract_global_common_pcs(
    checkpoint_path: Path | None,
    image_paths: List[Path],
    n_global_pcs: int,
    device: torch.device,
    batch_size: int = 32,
    model_type: str = "convnext",
) -> List[Dict[str, Any]]:
    """Extract global common PCs from all layers."""

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
            raise RuntimeError("open_clip_torch is required for CLIP extraction")

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

    extractor = ActivationExtractor(model, layer_names)

    # Extract GAP features from all layers
    print(f"Extracting GAP features from {len(target_layers)} layers...")

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
            if layer_name not in gap_features:
                gap_features[layer_name] = []
            gap_features[layer_name].append(gap)

        image_ids.extend(batch_ids)

    # Concatenate all batches
    for layer_name in gap_features:
        gap_features[layer_name] = np.vstack(gap_features[layer_name])

    extractor.remove_hooks()

    # Concatenate ALL layers horizontally for global PCA
    print(f"\nConcatenating features from {len(gap_features)} layers...")

    all_features = np.hstack([
        gap_features[layer]
        for layer in sorted(gap_features.keys())
    ])

    print(f"Concatenated feature shape: {all_features.shape}")
    print(f"  (n_images={all_features.shape[0]}, total_channels={all_features.shape[1]})")

    # Global PCA
    print(f"\nRunning Global PCA with {n_global_pcs} components...")
    n_components = min(n_global_pcs, all_features.shape[0], all_features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(all_features)

    global_scores = pca.transform(all_features)

    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Global PCs: {n_components} components explain {explained_var:.2f}% variance")

    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")

    # Build results
    results = []
    for img_idx, image_id in enumerate(image_ids):
        pc_scores = global_scores[img_idx].astype(np.float32)
        results.append({
            "image_id": image_id,
            "module_name": f"{model_type}.GlobalCommon",
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
        description="Extract Global Common PCs across all layers"
    )
    ap.add_argument("--convnext-checkpoint", type=Path, default=None)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data_images/dnn_pmods/orthogonalized"))
    ap.add_argument("--n-global-pcs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
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
        print("EXTRACTING CONVNEXT GLOBAL COMMON PCS")
        print("=" * 60)

        results = extract_global_common_pcs(
            args.convnext_checkpoint,
            image_paths,
            args.n_global_pcs,
            device,
            args.batch_size,
            model_type="convnext",
        )

        output_path = args.out_dir / "convnext_global_common_pcs.csv"
        save_results_to_csv(results, output_path, args.n_global_pcs)
        print(f"Saved: {output_path}")

    if run_clip:
        print("\n" + "=" * 60)
        print("EXTRACTING CLIP GLOBAL COMMON PCS")
        print("=" * 60)

        results = extract_global_common_pcs(
            None,
            image_paths,
            args.n_global_pcs,
            device,
            args.batch_size,
            model_type="clip",
        )

        output_path = args.out_dir / "clip_global_common_pcs.csv"
        save_results_to_csv(results, output_path, args.n_global_pcs)
        print(f"Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
