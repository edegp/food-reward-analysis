#!/usr/bin/env python3
"""
Extract activation-based PCA features from food images using ConvNeXt and CLIP models.

For each image, this script:
    1. Loads the image and applies model-specific preprocessing
    2. Passes the image through the model to extract layer activations
    3. Pools spatial activations to get a feature vector per layer
    4. Saves per-image, per-layer PC values (computed across the channel dimension)
    5. Adds a behavior-based rating layer (per-image mean rating across participants)

The output CSV format allows matching image-specific features with trial data
via ImageId (e.g., "0140" from rating_data CSV).

CSV format:
    - Columns: image_id, module_name, pc1, pc2, pc3, ...
    - One row per (image, layer) combination
    - image_id matches the "Image Name" column in rating_data*.csv

Usage (from repo root):
    uv run --project scripts/dnn_analysis/preprocess \
        python scripts/dnn_analysis/preprocess/extract_dnn_pcs.py \
        --convnext-checkpoint DNNs_model/v9/res_L/convnext_base_regression.pth \
        --image-dir Database \
        --out-dir data_images/dnn_pmods \
        --max-components 3
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from torchvision import transforms
from torchvision.models import convnext_base, vgg16
from tqdm import tqdm

try:
    import open_clip  # type: ignore
except Exception as exc:  # pragma: no cover
    open_clip = None
    _open_clip_error = exc
else:  # pragma: no cover
    _open_clip_error = None


def get_device() -> torch.device:
    """Get the best available device (MPS for Mac, CUDA for NVIDIA, CPU otherwise)."""

    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    else:
        print("Using CPU device")
        return torch.device("cpu")


def load_layer_groups(config_path: Path, model_type: str) -> Dict[str, List[str]]:
    """
    Load layer group configuration from layer_groups.json.

    Args:
        config_path: Path to layer_groups.json
        model_type: 'convnext' or 'clip'

    Returns:
        Dict mapping group labels to lists of layer names
        Example: {'Initial': ['features_1_0', 'features_1_1', ...], ...}
    """
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
    """Hook-based activation extractor for specified layers."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = set(layer_names)
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook function that saves activations."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and return captured activations."""
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.activations.copy()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def get_convnext_target_layers(model: nn.Module, prefix: str = "convnext.") -> List[Tuple[str, str]]:
    """
    Get target layer names for ConvNeXt.

    Returns:
        List of (full_name, display_name) tuples
        - GELU activations from each CNBlock (block.3)
        - Flatten layer output (classifier.1) - 1024-dim features before final linear
    """
    layers = []
    for name, module in model.named_modules():
        # Get GELU activation layers from CNBlocks
        if isinstance(module, nn.GELU) and name.endswith("block.4"):
            display_name = f"{prefix}{name.replace('.', '_')}"
            layers.append((name, display_name))
        # Get Flatten layer output (1024-dim features before final linear)
        elif isinstance(module, nn.Flatten) and name == "classifier.1":
            display_name = f"{prefix}{name.replace('.', '_')}"
            layers.append((name, display_name))

    return layers


def get_vgg_target_layers(model: nn.Module, prefix: str = "vgg.") -> List[Tuple[str, str]]:
    """
    Get target layer names for VGG16.

    Returns:
        List of (full_name, display_name) tuples
        - ReLU activations after each conv layer
        - Classifier FC layers' ReLU activations
    """
    layers = []

    # Track ReLU layers in features (after each conv layer)
    relu_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) and name.startswith("features"):
            display_name = f"{prefix}features_relu{relu_count}"
            layers.append((name, display_name))
            relu_count += 1
        # Get classifier ReLU layers (after FC1 and FC2)
        elif isinstance(module, nn.ReLU) and name.startswith("classifier"):
            display_name = f"{prefix}{name.replace('.', '_')}"
            layers.append((name, display_name))

    # Also add the avgpool output (before classifier)
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d) and name == "avgpool":
            display_name = f"{prefix}avgpool"
            layers.append((name, display_name))

    return layers


def get_clip_convnext_target_layers(model: nn.Module, prefix: str = "clip.") -> List[Tuple[str, str]]:
    """
    Get target layer names for CLIP ConvNeXt (timm-based).

    Returns:
        List of (full_name, display_name) tuples
        - GELU activations from each ConvNeXt block (mlp.act)
        - Head output (before final classification)
    """
    layers = []

    # Get trunk (feature extractor)
    trunk = getattr(model, 'trunk', model)  # Handle both TimmModel and direct trunk

    # Extract GELU from each stage's blocks
    if hasattr(trunk, 'stages'):
        for stage_idx, stage in enumerate(trunk.stages):
            if hasattr(stage, 'blocks'):
                for block_idx in range(len(stage.blocks)):
                    # Construct the full module path
                    full_name = f"trunk.stages.{stage_idx}.blocks.{block_idx}.mlp.act"
                    display_name = f"{prefix}stage{stage_idx}_{block_idx}"
                    layers.append((full_name, display_name))

    # Get head output (pooled features before final linear)
    if hasattr(model, 'head'):
        # Head processes trunk output, we'll capture it as a separate layer
        display_name = f"{prefix}head"
        layers.append(("head", display_name))

    return layers


def apply_pca(
    features: np.ndarray,
    pca: IncrementalPCA,
    max_components: int,
) -> np.ndarray:
    """
    Apply PCA transform and pad to max_components.

    Args:
        features: Shape (n_samples, n_features)
        pca: Fitted IncrementalPCA model
        max_components: Number of components to pad to

    Returns:
        PC scores padded to max_components: Shape (n_samples, max_components)
    """
    transformed = pca.transform(features.astype(np.float64))
    actual_components = transformed.shape[1]

    if actual_components >= max_components:
        return transformed[:, :max_components].astype(np.float32)

    # Pad with zeros
    n_samples = transformed.shape[0]
    result = np.zeros((n_samples, max_components), dtype=np.float32)
    result[:, :actual_components] = transformed.astype(np.float32)
    return result


def flatten_activation(activation: torch.Tensor) -> np.ndarray:
    """
    Flatten activation tensor without averaging.

    Args:
        activation: Shape (batch, channels, *spatial_dims) or (batch, features)

    Returns:
        Flattened features: Shape (batch, channels * spatial_size)
    """
    if activation.ndim == 2:
        # Already flattened (e.g., classifier output)
        return activation.cpu().numpy()
    elif activation.ndim >= 3:
        # Flatten all dimensions after batch: (B, C, H, W, ...) → (B, C*H*W*...)
        batch_size = activation.shape[0]
        flattened = activation.reshape(batch_size, -1)
        return flattened.cpu().numpy()
    else:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")


def normalize_image_id(value: Any) -> str | None:
    """Normalize image IDs to zero-padded 4-digit strings when possible."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].replace(".", "").isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(4)
    return text


def compute_mean_behavior_ratings(behavior_dir: Path) -> Dict[str, float]:
    """
    Compute per-image mean ratings across all subjects.
    """
    if not behavior_dir.exists():
        print(f"Warning: Behavior directory not found: {behavior_dir}")
        return {}

    rating_files = sorted(behavior_dir.glob("sub-*/rating_data*.csv"))
    rating_files += [p for p in behavior_dir.glob("*/rating_data*.csv") if p not in rating_files]

    if not rating_files:
        print(f"Warning: No rating_data CSV files found under {behavior_dir}")
        return {}

    rating_sums: Dict[str, float] = defaultdict(float)
    rating_counts: Dict[str, int] = defaultdict(int)

    column_aliases = {
        "image_name": {"image name", "imagename", "image_name", "image"},
        "rating": {"rating", "rating value", "rating_value"},
    }

    def find_column(columns: List[str], targets: set[str]) -> str | None:
        for col in columns:
            normalized = col.strip().lower()
            if normalized in targets:
                return col
        return None

    for csv_path in rating_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Warning: Failed to read {csv_path}: {exc}")
            continue

        image_col = find_column(df.columns.tolist(), column_aliases["image_name"])
        rating_col = find_column(df.columns.tolist(), column_aliases["rating"])

        if image_col is None or rating_col is None:
            print(f"Warning: Skipping {csv_path} (missing image or rating column)")
            continue

        ratings = pd.to_numeric(df[rating_col], errors="coerce")
        image_ids = df[image_col]

        mask = ratings.notna() & (ratings != 0)
        ratings = ratings[mask]
        image_ids = image_ids[mask]

        for img_value, rating in zip(image_ids, ratings):
            image_id = normalize_image_id(img_value)
            if image_id is None:
                continue
            rating_sums[image_id] += float(rating)
            rating_counts[image_id] += 1

    rating_means = {
        image_id: rating_sums[image_id] / rating_counts[image_id]
        for image_id in rating_sums
        if rating_counts[image_id] > 0
    }

    print(f"Computed behavior ratings for {len(rating_means)} images from {len(rating_files)} files")
    return rating_means


def append_rating_layer(
    result_rows: List[Dict[str, Any]],
    rating_means: Dict[str, float],
    max_components: int,
    module_name: str,
    image_ids_override: Optional[Iterable[str]] = None,
) -> None:
    """Append behavior rating averages as an extra synthetic layer."""
    if not rating_means:
        print(f"Warning: No behavior ratings to append for {module_name}")
        return

    if image_ids_override is not None:
        image_ids = [img_id for img_id in image_ids_override if img_id]
    else:
        image_ids = sorted({item["image_id"] for item in result_rows})

    if not image_ids:
        print(f"Warning: No image IDs available to append ratings for {module_name}")
        return

    existing_pairs = {(item["image_id"], item["module_name"]) for item in result_rows}

    appended = 0
    missing = 0

    for raw_id in image_ids:
        lookup_id = normalize_image_id(raw_id) or raw_id
        rating = rating_means.get(lookup_id)
        if rating is None:
            missing += 1
            continue

        key = (raw_id, module_name)
        if key in existing_pairs:
            continue

        pc_scores = np.zeros((max_components,), dtype=np.float32)
        pc_scores[0] = np.float32(rating)

        result_rows.append({
            "image_id": raw_id,
            "module_name": module_name,
            "pc_scores": pc_scores,
        })
        appended += 1
        existing_pairs.add(key)

    print(f"Appended {module_name}: {appended} images with ratings, {missing} missing ratings")


def load_image_batch(
    image_paths: List[Path],
    preprocess: transforms.Compose,
    start_idx: int,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Load and preprocess a batch of images."""
    batch_images = []
    batch_ids = []

    end_idx = min(start_idx + batch_size, len(image_paths))

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
        return torch.empty(0), []

    return torch.stack(batch_images), batch_ids


@torch.no_grad()
def extract_convnext_activations(
    checkpoint_path: Path,
    image_paths: List[Path],
    max_components: int,
    device: torch.device,
    batch_size: int = 32,
    temp_dir: Path = None,
) -> List[Dict[str, Any]]:
    """Extract activation-based PCA features from ConvNeXt model."""
    # Load model
    model = convnext_base(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned_state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    # Get target layers
    target_layers = get_convnext_target_layers(model)
    layer_names = [name for name, _ in target_layers]
    name_map = {name: display for name, display in target_layers}

    # Setup activation extractor
    extractor = ActivationExtractor(model, layer_names)

    # Preprocessing (ConvNeXt ImageNet normalization)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Step 1: Extract all activations and save to disk
    import shutil

    if temp_dir is None:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
    else:
        temp_dir = temp_dir / "convnext_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

    try:
        print(f"Step 1: Extracting activations from {len(image_paths)} images (batch_size={batch_size})...")
        image_ids = []

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
            start_idx = batch_idx * batch_size
            batch_tensor, batch_ids = load_image_batch(image_paths, preprocess, start_idx, batch_size)

            if batch_tensor.numel() == 0:
                continue

            batch_tensor = batch_tensor.to(device)

            try:
                activations = extractor(batch_tensor)

                # Save each layer's activations to disk
                for layer_name, activation in activations.items():
                    flattened = flatten_activation(activation)  # (batch, channels * spatial)

                    # Save to disk with batch index
                    layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{batch_idx}.npy"

                    if batch_idx == 0:  # Only print debug info for first batch
                        print(f"  DEBUG: {layer_name} - activation shape: {activation.shape}, flattened shape: {flattened.shape}, size: {flattened.nbytes / 1024 / 1024:.2f} MB")

                    try:
                        np.save(layer_file, flattened)

                        # Verify the file was written correctly
                        if not layer_file.exists():
                            raise IOError(f"File {layer_file} was not created")

                        file_size = layer_file.stat().st_size
                        if batch_idx == 0:
                            print(f"    DEBUG: File size: {file_size / 1024 / 1024:.2f} MB")

                        if file_size < 1000:  # Less than 1KB is suspicious
                            raise IOError(f"File {layer_file} is only {file_size} bytes - likely failed to write")

                    except Exception as e:
                        print(f"    ERROR writing {layer_file}: {e}")
                        raise

                image_ids.extend(batch_ids)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        # Step 2: Process each layer - load from disk, fit PCA, apply PCA
        print(f"\nStep 2: Fitting PCA for {len(layer_names)} layers...")
        results = []

        for layer_idx, layer_name in enumerate(layer_names):
            display_name = name_map[layer_name]
            print(f"\nProcessing layer {layer_idx + 1}/{len(layer_names)}: {display_name}")

            # Load all batches for this layer from disk
            layer_features = []
            layer_file_pattern = f"{layer_name.replace('.', '_')}_batch*.npy"
            layer_files = sorted(temp_dir.glob(layer_file_pattern))

            print(f"  Found {len(layer_files)} batch files for {display_name}")

            for idx, layer_file in enumerate(layer_files):
                file_size = layer_file.stat().st_size
                print(f"    Loading batch {idx}: {layer_file.name}, size: {file_size / 1024 / 1024:.2f} MB")

                try:
                    batch_features = np.load(layer_file)
                    print(f"      Loaded shape: {batch_features.shape}")

                    for i in range(batch_features.shape[0]):
                        layer_features.append(batch_features[i])

                except Exception as e:
                    print(f"      ERROR loading {layer_file}: {e}")
                    print(f"      File exists: {layer_file.exists()}, size: {file_size}")
                    raise

            # Fit PCA for this layer
            if len(layer_features) >= 2:
                n_samples = len(layer_features)
                n_features = layer_features[0].shape[0]
                n_components = min(max_components, n_samples, n_features)

                if n_components >= 1:
                    pca = IncrementalPCA(n_components=n_components, batch_size=min(100, n_samples))

                    try:
                        # Fit in batches
                        pca_batch_size = min(100, n_samples)
                        for i in range(0, n_samples, pca_batch_size):
                            batch = np.array(layer_features[i:i + pca_batch_size])
                            pca.partial_fit(batch.astype(np.float64))

                        print(f"  Fitted PCA: {n_samples} images, {n_features} features → {n_components} components")

                        # Apply PCA to each image for this layer
                        for img_idx, image_id in enumerate(image_ids):
                            features = layer_features[img_idx].reshape(1, -1)
                            pc_scores = apply_pca(features, pca, max_components)[0]

                            results.append({
                                "image_id": image_id,
                                "module_name": display_name,
                                "pc_scores": pc_scores,
                            })

                    except Exception as e:
                        print(f"  PCA fit/apply failed: {e}")

            # Free memory
            del layer_features

    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    extractor.remove_hooks()
    return results


@torch.no_grad()
def extract_convnext_activations_group_pca(
    checkpoint_path: Path,
    image_paths: List[Path],
    layer_groups_config: Path,
    max_components_per_layer: int,
    device: torch.device,
    batch_size: int = 32,
    temp_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Extract LayerGroup-based PCA features from ConvNeXt model.

    Args:
        checkpoint_path: Path to ConvNeXt checkpoint
        image_paths: List of image paths
        layer_groups_config: Path to layer groups JSON config
        max_components_per_layer: Number of PCA components per layer
        device: torch device
        batch_size: Batch size for feature extraction
        temp_dir: Optional temporary directory for storing activations
    """
    # Load layer groups configuration
    layer_groups = load_layer_groups(layer_groups_config, "convnext")

    print(f"Loaded {len(layer_groups)} layer groups: {list(layer_groups.keys())}")

    # Load trained model
    model = convnext_base(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned_state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    # Get target layers
    target_layers = get_convnext_target_layers(model)
    layer_names = [name for name, _ in target_layers]
    name_map = {name: display for name, display in target_layers}

    # Setup activation extractor
    extractor = ActivationExtractor(model, layer_names)

    # Preprocessing (ConvNeXt ImageNet normalization)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Setup temporary directory
    if temp_dir is None:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
    else:
        temp_dir = temp_dir / "convnext_group_pca_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

    try:
        # Step 1: Extract all activations and save to disk
        print(f"\nStep 1: Extracting activations from {len(image_paths)} images (batch_size={batch_size})...")
        image_ids = []

        for batch_idx in tqdm(range(0, len(image_paths), batch_size), desc="Extracting activations"):
            batch_paths = image_paths[batch_idx:batch_idx + batch_size]
            batch_tensors = []

            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img)
                batch_tensors.append(img_tensor)
                image_ids.append(img_path.stem)

            batch_input = torch.stack(batch_tensors).to(device)

            # Extract activations
            activations = extractor(batch_input)

            # Save activations for each layer
            for layer_name, activation in activations.items():
                flattened = flatten_activation(activation)  # (batch, features)
                save_path = temp_dir / f"{layer_name.replace('.', '_')}_batch{batch_idx // batch_size}.npy"
                np.save(save_path, flattened)

        print(f"Extracted activations for {len(image_ids)} images")

        # Calculate global max_components across all groups
        max_layers_in_any_group = max(len(group_layers) for group_layers in layer_groups.values())
        global_max_components = max_layers_in_any_group * max_components_per_layer
        print(f"Using global max components: {global_max_components} ({max_layers_in_any_group} layers × {max_components_per_layer})")

        results = []
        reverse_name_map = {v: k for k, v in name_map.items()}
        pca_fit_batch_size = 50  # Read 50 images at a time for PCA fitting
        n_images = len(image_ids)

        # Process each group separately
        for group_idx, (group_label, group_layers) in enumerate(layer_groups.items(), 1):
            print(f"\nProcessing group {group_idx}/{len(layer_groups)}: {group_label} ({len(group_layers)} layers)")

            # Get layer names for this group
            group_layer_names = []
            for layer_display in group_layers:
                layer_name = reverse_name_map.get(layer_display)
                if layer_name and layer_name in layer_names:
                    group_layer_names.append(layer_name)

            if not group_layer_names:
                print(f"  Warning: No valid layers found for group {group_label}")
                continue

            # Get batch file info
            layer_file_pattern = f"{group_layer_names[0].replace('.', '_')}_batch*.npy"
            batch_files = sorted(temp_dir.glob(layer_file_pattern))
            n_batches = len(batch_files)

            # Step 2: Fit PCA using IncrementalPCA (read batches incrementally)
            print(f"  Fitting Group PCA on {n_images} images...")

            # Determine n_components
            n_components = len(group_layer_names) * max_components_per_layer

            # First pass: get feature dimension
            first_batch_features = []
            for layer_name in group_layer_names:
                layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch0.npy"
                feat = np.load(layer_file)[0]  # First image
                first_batch_features.append(feat)
            concatenated_sample = np.concatenate(first_batch_features)
            n_features = concatenated_sample.shape[0]

            # Adjust n_components if necessary
            n_components = min(n_components, n_images, n_features)

            print(f"=== Processing group: {group_label} ===")
            print(f"Layers in group: {len(group_layer_names)}")
            print(f"Group {group_label}: {n_images} images, {n_features} features → {n_components} components")

            # Initialize IncrementalPCA
            from sklearn.decomposition import IncrementalPCA
            pca_model = IncrementalPCA(n_components=n_components, batch_size=min(100, n_images))

            # Fit PCA incrementally by reading batches
            images_processed = 0
            for read_batch_start in range(0, n_images, pca_fit_batch_size):
                read_batch_end = min(read_batch_start + pca_fit_batch_size, n_images)

                # Load this batch of images' features
                batch_features_list = []
                for img_idx in range(read_batch_start, read_batch_end):
                    # Figure out which file this image is in
                    file_idx = img_idx // batch_size
                    within_file_idx = img_idx % batch_size

                    # Load features for this image from all layers in the group
                    img_features = []
                    for layer_name in group_layer_names:
                        layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{file_idx}.npy"
                        layer_batch = np.load(layer_file)
                        if within_file_idx < layer_batch.shape[0]:
                            img_features.append(layer_batch[within_file_idx])

                    if len(img_features) == len(group_layer_names):
                        concatenated = np.concatenate(img_features)
                        batch_features_list.append(concatenated)

                if batch_features_list:
                    batch_array = np.array(batch_features_list, dtype=np.float64)
                    pca_model.partial_fit(batch_array)
                    images_processed += len(batch_features_list)

            explained_var = np.sum(pca_model.explained_variance_ratio_) * 100
            print(f"Fitted PCA for {group_label}: {n_components} components explain {explained_var:.2f}% variance")

            # Step 3: Apply PCA to all images (read in batches)
            print(f"  Applying PCA to {n_images} images...")
            for read_batch_start in range(0, n_images, pca_fit_batch_size):
                read_batch_end = min(read_batch_start + pca_fit_batch_size, n_images)

                # Load this batch of images' features
                for img_idx in range(read_batch_start, read_batch_end):
                    # Figure out which file this image is in
                    file_idx = img_idx // batch_size
                    within_file_idx = img_idx % batch_size

                    # Load features for this image from all layers in the group
                    img_features = []
                    for layer_name in group_layer_names:
                        layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{file_idx}.npy"
                        layer_batch = np.load(layer_file)
                        if within_file_idx < layer_batch.shape[0]:
                            img_features.append(layer_batch[within_file_idx])

                    if len(img_features) != len(group_layer_names):
                        continue

                    # Concatenate and apply PCA
                    concatenated = np.concatenate(img_features).reshape(1, -1)

                    try:
                        transformed = pca_model.transform(concatenated.astype(np.float64))
                        actual_components = transformed.shape[1]

                        # Use global max_components for all groups
                        pc_scores = np.zeros(global_max_components, dtype=np.float32)
                        pc_scores[:actual_components] = transformed[0, :].astype(np.float32)

                        results.append({
                            "image_id": image_ids[img_idx],
                            "module_name": f"convnext.{group_label}",
                            "pc_scores": pc_scores,
                        })
                    except Exception as e:
                        print(f"  PCA transform failed for image {image_ids[img_idx]}: {e}")
                        continue

            print(f"  Completed group {group_label}")

            # Free memory
            del pca_model

        print(f"\nGenerated {len(results)} group-level PC score rows")

    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    extractor.remove_hooks()
    return results


@torch.no_grad()
def extract_clip_activations(
    image_paths: List[Path],
    max_components: int,
    device: torch.device,
    batch_size: int = 32,
    temp_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Extract activation-based PCA features from CLIP ConvNeXt-Base model.

    Args:
        temp_dir: Optional temporary directory for storing activations.
                  If None, uses system temp directory.
    """
    if open_clip is None:
        raise RuntimeError(
            "open_clip_torch is required for CLIP extraction"
            + (f": {type(_open_clip_error).__name__}: {_open_clip_error}" if _open_clip_error else "")
        )

    # Load CLIP ConvNeXt-Base model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "convnext_base",
        pretrained="laion400m_s13b_b51k"
    )
    visual = model.visual
    visual.to(device)
    visual.eval()

    # Get target layers for ConvNeXt
    target_layers = get_clip_convnext_target_layers(visual)
    layer_names = [name for name, _ in target_layers]
    name_map = {name: display for name, display in target_layers}

    # Setup activation extractor
    extractor = ActivationExtractor(visual, layer_names)

    # Step 1: Extract all activations and save to disk
    import tempfile
    import shutil

    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    else:
        temp_dir = temp_dir / "clip_per_layer_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using temporary directory: {temp_dir}")

    try:
        print(f"Step 1: Extracting activations from {len(image_paths)} images (batch_size={batch_size})...")
        image_ids = []

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
            start_idx = batch_idx * batch_size
            batch_tensor, batch_ids = load_image_batch(image_paths, preprocess, start_idx, batch_size)

            if batch_tensor.numel() == 0:
                continue

            batch_tensor = batch_tensor.to(device)

            try:
                activations = extractor(batch_tensor)

                # Save each layer's activations to disk
                for layer_name, activation in activations.items():
                    flattened = flatten_activation(activation)  # (batch, channels * spatial)

                    # Save to disk with batch index
                    layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{batch_idx}.npy"

                    if batch_idx == 0:  # Only print debug info for first batch
                        print(f"  DEBUG: {layer_name} - activation shape: {activation.shape}, flattened shape: {flattened.shape}, size: {flattened.nbytes / 1024 / 1024:.2f} MB")

                    try:
                        np.save(layer_file, flattened)

                        # Verify the file was written correctly
                        if not layer_file.exists():
                            raise IOError(f"File {layer_file} was not created")

                        file_size = layer_file.stat().st_size
                        if batch_idx == 0:
                            print(f"    DEBUG: File size: {file_size / 1024 / 1024:.2f} MB")

                        if file_size < 1000:  # Less than 1KB is suspicious
                            raise IOError(f"File {layer_file} is only {file_size} bytes - likely failed to write")

                    except Exception as e:
                        print(f"    ERROR writing {layer_file}: {e}")
                        raise

                image_ids.extend(batch_ids)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        # Step 2: Process each layer - load from disk, fit PCA, apply PCA
        print(f"\nStep 2: Fitting PCA for {len(layer_names)} layers...")
        results = []

        for layer_idx, layer_name in enumerate(layer_names):
            display_name = name_map[layer_name]
            print(f"\nProcessing layer {layer_idx + 1}/{len(layer_names)}: {display_name}")

            # Load all batches for this layer from disk
            layer_features = []
            layer_file_pattern = f"{layer_name.replace('.', '_')}_batch*.npy"
            layer_files = sorted(temp_dir.glob(layer_file_pattern))

            print(f"  Found {len(layer_files)} batch files for {display_name}")

            for idx, layer_file in enumerate(layer_files):
                file_size = layer_file.stat().st_size
                print(f"    Loading batch {idx}: {layer_file.name}, size: {file_size / 1024 / 1024:.2f} MB")

                try:
                    batch_features = np.load(layer_file)
                    print(f"      Loaded shape: {batch_features.shape}")

                    for i in range(batch_features.shape[0]):
                        layer_features.append(batch_features[i])

                except Exception as e:
                    print(f"      ERROR loading {layer_file}: {e}")
                    print(f"      File exists: {layer_file.exists()}, size: {file_size}")
                    raise

            # Fit PCA for this layer
            if len(layer_features) >= 2:
                n_samples = len(layer_features)
                n_features = layer_features[0].shape[0]
                n_components = min(max_components, n_samples, n_features)

                if n_components >= 1:
                    pca = IncrementalPCA(n_components=n_components, batch_size=min(100, n_samples))

                    try:
                        # Fit in batches
                        pca_batch_size = min(100, n_samples)
                        for i in range(0, n_samples, pca_batch_size):
                            batch = np.array(layer_features[i:i + pca_batch_size])
                            pca.partial_fit(batch.astype(np.float64))

                        print(f"  Fitted PCA: {n_samples} images, {n_features} features → {n_components} components")

                        # Apply PCA to each image for this layer
                        for img_idx, image_id in enumerate(image_ids):
                            features = layer_features[img_idx].reshape(1, -1)
                            pc_scores = apply_pca(features, pca, max_components)[0]

                            results.append({
                                "image_id": image_id,
                                "module_name": display_name,
                                "pc_scores": pc_scores,
                            })

                    except Exception as e:
                        print(f"  PCA fit/apply failed: {e}")

            # Free memory
            del layer_features

    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    extractor.remove_hooks()
    return results


@torch.no_grad()
def extract_clip_activations_group_pca(
    image_paths: List[Path],
    layer_groups_config: Path,
    max_components_per_layer: int = 2,
    device: torch.device = None,
    batch_size: int = 32,
    temp_dir: Path = None,
) -> List[Dict[str, Any]]:
    """
    Extract activation-based Group PCA features from CLIP ConvNeXt-Base model.

    Instead of per-layer PCA, this fits PCA per LayerGroup with
    n_components = num_layers_in_group * max_components_per_layer.

    Args:
        image_paths: List of image file paths
        layer_groups_config: Path to layer_groups.json
        max_components_per_layer: Number of PC components per layer in group (default: 2)
        device: Torch device (if None, auto-detect)
        batch_size: Batch size for image processing
        temp_dir: Temporary directory for saving activations

    Returns:
        List of dicts with keys: image_id, module_name (group label), pc_scores
    """
    if device is None:
        device = get_device()

    if open_clip is None:
        raise RuntimeError(
            "open_clip_torch is required for CLIP extraction"
            + (f": {type(_open_clip_error).__name__}: {_open_clip_error}" if _open_clip_error else "")
        )

    # Load CLIP ConvNeXt-Base model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "convnext_base",
        pretrained="laion400m_s13b_b51k"
    )
    visual = model.visual
    visual.to(device)
    visual.eval()

    # Get target layers for ConvNeXt
    target_layers = get_clip_convnext_target_layers(visual)
    layer_names = [name for name, _ in target_layers]
    name_map = {name: display for name, display in target_layers}

    # Load layer groups
    layer_groups = load_layer_groups(layer_groups_config, "clip")
    print(f"Loaded {len(layer_groups)} layer groups: {list(layer_groups.keys())}")

    # Setup activation extractor
    extractor = ActivationExtractor(visual, layer_names)

    # Step 1: Extract all activations and save to disk
    import tempfile
    import shutil

    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    else:
        temp_dir = temp_dir / "clip_group_pca_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

    try:
        print(f"\nStep 1: Extracting activations from {len(image_paths)} images (batch_size={batch_size})...")
        image_ids = []

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
            start_idx = batch_idx * batch_size
            batch_tensor, batch_ids = load_image_batch(image_paths, preprocess, start_idx, batch_size)

            if batch_tensor.numel() == 0:
                continue

            batch_tensor = batch_tensor.to(device)

            try:
                activations = extractor(batch_tensor)

                # Save each layer's activations to disk
                for layer_name, activation in activations.items():
                    flattened = flatten_activation(activation)  # (batch, features)
                    layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{batch_idx}.npy"
                    np.save(layer_file, flattened)

                image_ids.extend(batch_ids)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        print(f"Extracted activations for {len(image_ids)} images")

        # Calculate global max_components across all groups
        max_layers_in_any_group = max(len(group_layers) for group_layers in layer_groups.values())
        global_max_components = max_layers_in_any_group * max_components_per_layer
        print(f"Using global max components: {global_max_components} ({max_layers_in_any_group} layers × {max_components_per_layer})")

        results = []
        reverse_name_map = {v: k for k, v in name_map.items()}
        pca_fit_batch_size = 50  # Read 50 images at a time for PCA fitting
        n_images = len(image_ids)

        # Process each group separately
        for group_idx, (group_label, group_layers) in enumerate(layer_groups.items(), 1):
            print(f"\nProcessing group {group_idx}/{len(layer_groups)}: {group_label} ({len(group_layers)} layers)")

            # Get layer names for this group
            group_layer_names = []
            for layer_display in group_layers:
                layer_name = reverse_name_map.get(layer_display)
                if layer_name and layer_name in layer_names:
                    group_layer_names.append(layer_name)

            if not group_layer_names:
                print(f"  Warning: No valid layers found for group {group_label}")
                continue

            # Get batch file info
            layer_file_pattern = f"{group_layer_names[0].replace('.', '_')}_batch*.npy"
            batch_files = sorted(temp_dir.glob(layer_file_pattern))
            n_batches = len(batch_files)

            # Step 2: Fit PCA using IncrementalPCA (read batches incrementally)
            print(f"  Fitting Group PCA on {n_images} images...")

            # Determine n_components
            n_components = len(group_layer_names) * max_components_per_layer

            # First pass: get feature dimension
            first_batch_features = []
            for layer_name in group_layer_names:
                layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch0.npy"
                feat = np.load(layer_file)[0]  # First image
                first_batch_features.append(feat)
            concatenated_sample = np.concatenate(first_batch_features)
            n_features = concatenated_sample.shape[0]

            # Adjust n_components if necessary
            n_components = min(n_components, n_images, n_features)

            print(f"=== Processing group: {group_label} ===")
            print(f"Layers in group: {len(group_layer_names)}")
            print(f"Group {group_label}: {n_images} images, {n_features} features → {n_components} components")

            # Initialize IncrementalPCA
            from sklearn.decomposition import IncrementalPCA
            pca_model = IncrementalPCA(n_components=n_components, batch_size=min(100, n_images))

            # Fit PCA incrementally by reading batches
            for read_batch_start in range(0, n_images, pca_fit_batch_size):
                read_batch_end = min(read_batch_start + pca_fit_batch_size, n_images)

                # Load this batch of images' features
                batch_features_list = []
                for img_idx in range(read_batch_start, read_batch_end):
                    # Figure out which file this image is in
                    file_idx = img_idx // batch_size
                    within_file_idx = img_idx % batch_size

                    # Load features for this image from all layers in the group
                    img_features = []
                    for layer_name in group_layer_names:
                        layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{file_idx}.npy"
                        layer_batch = np.load(layer_file)
                        if within_file_idx < layer_batch.shape[0]:
                            img_features.append(layer_batch[within_file_idx])

                    if len(img_features) == len(group_layer_names):
                        concatenated = np.concatenate(img_features)
                        batch_features_list.append(concatenated)

                if batch_features_list:
                    batch_array = np.array(batch_features_list, dtype=np.float64)
                    pca_model.partial_fit(batch_array)

            explained_var = np.sum(pca_model.explained_variance_ratio_) * 100
            print(f"Fitted PCA for {group_label}: {n_components} components explain {explained_var:.2f}% variance")

            # Step 3: Apply PCA to all images (read in batches)
            print(f"  Applying PCA to {n_images} images...")
            for read_batch_start in range(0, n_images, pca_fit_batch_size):
                read_batch_end = min(read_batch_start + pca_fit_batch_size, n_images)

                # Load this batch of images' features
                for img_idx in range(read_batch_start, read_batch_end):
                    # Figure out which file this image is in
                    file_idx = img_idx // batch_size
                    within_file_idx = img_idx % batch_size

                    # Load features for this image from all layers in the group
                    img_features = []
                    for layer_name in group_layer_names:
                        layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{file_idx}.npy"
                        layer_batch = np.load(layer_file)
                        if within_file_idx < layer_batch.shape[0]:
                            img_features.append(layer_batch[within_file_idx])

                    if len(img_features) != len(group_layer_names):
                        continue

                    # Concatenate and apply PCA
                    concatenated = np.concatenate(img_features).reshape(1, -1)

                    try:
                        transformed = pca_model.transform(concatenated.astype(np.float64))
                        actual_components = transformed.shape[1]

                        # Use global max_components for all groups
                        pc_scores = np.zeros(global_max_components, dtype=np.float32)
                        pc_scores[:actual_components] = transformed[0, :].astype(np.float32)

                        results.append({
                            "image_id": image_ids[img_idx],
                            "module_name": f"clip.{group_label}",
                            "pc_scores": pc_scores,
                        })
                    except Exception as e:
                        print(f"  PCA transform failed for image {image_ids[img_idx]}: {e}")
                        continue

            print(f"  Completed group {group_label}")

            # Free memory
            del pca_model

        print(f"\nGenerated {len(results)} group-level PC score rows")

    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    extractor.remove_hooks()
    return results


@torch.no_grad()
def extract_vgg_activations(
    checkpoint_path: Path,
    image_paths: List[Path],
    max_components: int,
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Extract activation-based PCA features from VGG16 model with custom checkpoint."""
    # Load VGG16 with custom checkpoint
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)  # Regression output
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned_state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    # Get target layers
    target_layers = get_vgg_target_layers(model)
    layer_names = [name for name, _ in target_layers]
    name_map = {name: display for name, display in target_layers}

    # Setup activation extractor
    extractor = ActivationExtractor(model, layer_names)

    # Preprocessing (VGG ImageNet normalization)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Step 1: Extract all activations and save to disk
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")

    try:
        print(f"Step 1: Extracting activations from {len(image_paths)} images (batch_size={batch_size})...")
        image_ids = []

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Extracting VGG activations"):
            start_idx = batch_idx * batch_size
            batch_tensor, batch_ids = load_image_batch(image_paths, preprocess, start_idx, batch_size)

            if batch_tensor.numel() == 0:
                continue

            batch_tensor = batch_tensor.to(device)

            try:
                activations = extractor(batch_tensor)

                # Save each layer's activations to disk
                for layer_name, activation in activations.items():
                    flattened = flatten_activation(activation)

                    layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{batch_idx}.npy"

                    if batch_idx == 0:
                        print(f"  DEBUG: {layer_name} - activation shape: {activation.shape}, flattened shape: {flattened.shape}")

                    np.save(layer_file, flattened)

                image_ids.extend(batch_ids)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        # Step 2: Process each layer - load from disk, fit PCA, apply PCA
        print(f"\nStep 2: Fitting PCA for {len(layer_names)} layers...")
        results = []

        for layer_idx, layer_name in enumerate(layer_names):
            display_name = name_map[layer_name]
            print(f"\nProcessing layer {layer_idx + 1}/{len(layer_names)}: {display_name}")

            # Load all batches for this layer from disk
            layer_features = []
            layer_file_pattern = f"{layer_name.replace('.', '_')}_batch*.npy"
            layer_files = sorted(temp_dir.glob(layer_file_pattern))

            for layer_file in layer_files:
                try:
                    batch_features = np.load(layer_file)
                    for i in range(batch_features.shape[0]):
                        layer_features.append(batch_features[i])
                except Exception as e:
                    print(f"  ERROR loading {layer_file}: {e}")
                    raise

            # Fit PCA for this layer
            if len(layer_features) >= 2:
                n_samples = len(layer_features)
                n_features = layer_features[0].shape[0]
                n_components = min(max_components, n_samples, n_features)

                if n_components >= 1:
                    pca = IncrementalPCA(n_components=n_components, batch_size=min(100, n_samples))

                    try:
                        pca_batch_size = min(100, n_samples)
                        for i in range(0, n_samples, pca_batch_size):
                            batch = np.array(layer_features[i:i + pca_batch_size])
                            pca.partial_fit(batch.astype(np.float64))

                        print(f"  Fitted PCA: {n_samples} images, {n_features} features → {n_components} components")

                        for img_idx, image_id in enumerate(image_ids):
                            features = layer_features[img_idx].reshape(1, -1)
                            pc_scores = apply_pca(features, pca, max_components)[0]

                            results.append({
                                "image_id": image_id,
                                "module_name": display_name,
                                "pc_scores": pc_scores,
                            })

                    except Exception as e:
                        print(f"  PCA fit/apply failed: {e}")

            del layer_features

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    extractor.remove_hooks()
    return results


@torch.no_grad()
def extract_vgg_activations_group_pca(
    checkpoint_path: Path,
    image_paths: List[Path],
    layer_groups_config: Path,
    max_components_per_layer: int = 2,
    device: torch.device = None,
    batch_size: int = 32,
    temp_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Extract activation-based Group PCA features from VGG16 model with custom checkpoint.
    """
    if device is None:
        device = get_device()

    # Load VGG16 with custom checkpoint
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)  # Regression output
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned_state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    # Get target layers
    target_layers = get_vgg_target_layers(model)
    layer_names = [name for name, _ in target_layers]
    name_map = {name: display for name, display in target_layers}

    # Load layer groups
    layer_groups = load_layer_groups(layer_groups_config, "vgg")
    print(f"Loaded {len(layer_groups)} layer groups: {list(layer_groups.keys())}")

    # Setup activation extractor
    extractor = ActivationExtractor(model, layer_names)

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    import shutil

    if temp_dir is None:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
    else:
        temp_dir = temp_dir / "vgg_group_pca_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

    try:
        print(f"\nStep 1: Extracting activations from {len(image_paths)} images (batch_size={batch_size})...")
        image_ids = []

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Extracting VGG activations"):
            start_idx = batch_idx * batch_size
            batch_tensor, batch_ids = load_image_batch(image_paths, preprocess, start_idx, batch_size)

            if batch_tensor.numel() == 0:
                continue

            batch_tensor = batch_tensor.to(device)

            try:
                activations = extractor(batch_tensor)

                for layer_name, activation in activations.items():
                    flattened = flatten_activation(activation)
                    layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{batch_idx}.npy"
                    np.save(layer_file, flattened)

                image_ids.extend(batch_ids)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        print(f"Extracted activations for {len(image_ids)} images")

        # Calculate global max_components across all groups
        max_layers_in_any_group = max(len(group_layers) for group_layers in layer_groups.values())
        global_max_components = max_layers_in_any_group * max_components_per_layer
        print(f"Using global max components: {global_max_components}")

        results = []
        reverse_name_map = {v: k for k, v in name_map.items()}
        pca_fit_batch_size = 50
        n_images = len(image_ids)

        # Process each group separately
        for group_idx, (group_label, group_layers) in enumerate(layer_groups.items(), 1):
            print(f"\nProcessing group {group_idx}/{len(layer_groups)}: {group_label} ({len(group_layers)} layers)")

            group_layer_names = []
            for layer_display in group_layers:
                layer_name = reverse_name_map.get(layer_display)
                if layer_name and layer_name in layer_names:
                    group_layer_names.append(layer_name)

            if not group_layer_names:
                print(f"  Warning: No valid layers found for group {group_label}")
                continue

            # Determine n_components
            n_components = len(group_layer_names) * max_components_per_layer

            # Get feature dimension from first batch
            first_batch_features = []
            for layer_name in group_layer_names:
                layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch0.npy"
                feat = np.load(layer_file)[0]
                first_batch_features.append(feat)
            concatenated_sample = np.concatenate(first_batch_features)
            n_features = concatenated_sample.shape[0]

            n_components = min(n_components, n_images, n_features)

            print(f"Group {group_label}: {n_images} images, {n_features} features → {n_components} components")

            # Initialize IncrementalPCA
            pca_model = IncrementalPCA(n_components=n_components, batch_size=min(100, n_images))

            # Fit PCA incrementally
            for read_batch_start in range(0, n_images, pca_fit_batch_size):
                read_batch_end = min(read_batch_start + pca_fit_batch_size, n_images)

                batch_features_list = []
                for img_idx in range(read_batch_start, read_batch_end):
                    file_idx = img_idx // batch_size
                    within_file_idx = img_idx % batch_size

                    img_features = []
                    for layer_name in group_layer_names:
                        layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{file_idx}.npy"
                        layer_batch = np.load(layer_file)
                        if within_file_idx < layer_batch.shape[0]:
                            img_features.append(layer_batch[within_file_idx])

                    if len(img_features) == len(group_layer_names):
                        concatenated = np.concatenate(img_features)
                        batch_features_list.append(concatenated)

                if batch_features_list:
                    batch_array = np.array(batch_features_list, dtype=np.float64)
                    pca_model.partial_fit(batch_array)

            explained_var = np.sum(pca_model.explained_variance_ratio_) * 100
            print(f"Fitted PCA for {group_label}: {n_components} components explain {explained_var:.2f}% variance")

            # Apply PCA to all images
            print(f"  Applying PCA to {n_images} images...")
            for read_batch_start in range(0, n_images, pca_fit_batch_size):
                read_batch_end = min(read_batch_start + pca_fit_batch_size, n_images)

                for img_idx in range(read_batch_start, read_batch_end):
                    file_idx = img_idx // batch_size
                    within_file_idx = img_idx % batch_size

                    img_features = []
                    for layer_name in group_layer_names:
                        layer_file = temp_dir / f"{layer_name.replace('.', '_')}_batch{file_idx}.npy"
                        layer_batch = np.load(layer_file)
                        if within_file_idx < layer_batch.shape[0]:
                            img_features.append(layer_batch[within_file_idx])

                    if len(img_features) != len(group_layer_names):
                        continue

                    concatenated = np.concatenate(img_features).reshape(1, -1)

                    try:
                        transformed = pca_model.transform(concatenated.astype(np.float64))
                        actual_components = transformed.shape[1]

                        pc_scores = np.zeros(global_max_components, dtype=np.float32)
                        pc_scores[:actual_components] = transformed[0, :].astype(np.float32)

                        results.append({
                            "image_id": image_ids[img_idx],
                            "module_name": f"vgg.{group_label}",
                            "pc_scores": pc_scores,
                        })
                    except Exception as e:
                        print(f"  PCA transform failed for image {image_ids[img_idx]}: {e}")
                        continue

            print(f"  Completed group {group_label}")
            del pca_model

        print(f"\nGenerated {len(results)} group-level PC score rows")

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    extractor.remove_hooks()
    return results


def save_results_to_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
    max_components: int,
) -> None:
    """
    Save per-image, per-layer PC scores to CSV.

    CSV format:
        image_id, module_name, pc1, pc2, pc3, ...
    """
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
                row[f"pc{i+1}"] = float(item["pc_scores"][i])
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convnext-checkpoint", type=Path, default=None,
                    help="Path to ConvNeXt checkpoint (required if running ConvNeXt)")
    ap.add_argument("--image-dir", type=Path, required=True,
                    help="Directory containing food images (e.g., Database)")
    ap.add_argument("--out-dir", type=Path, default=Path("data_images/dnn_pmods"))
    ap.add_argument("--max-components", type=int, default=3,
                    help="Max PC components per layer (layer-level PCA mode)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for processing images (default: 64)")
    ap.add_argument("--behavior-dir", type=Path, default=Path("Food_Behavior"),
                    help="Directory containing subject rating_data*.csv files")
    ap.add_argument("--disable-convnext", action="store_true",
                    help="Skip ConvNeXt feature extraction")
    ap.add_argument("--disable-clip", action="store_true",
                    help="Skip CLIP feature extraction")
    ap.add_argument("--disable-vgg", action="store_true",
                    help="Skip VGG feature extraction")
    ap.add_argument("--enable-vgg", action="store_true",
                    help="Enable VGG feature extraction (disabled by default)")
    ap.add_argument("--vgg-checkpoint", type=Path, default=None,
                    help="Path to VGG checkpoint (required if running VGG)")

    # Group PCA options
    ap.add_argument("--use-group-pca", action="store_true",
                    help="Use LayerGroup-based PCA instead of per-layer PCA")
    ap.add_argument("--layer-groups-config", type=Path,
                    default=Path("scripts/dnn_analysis/config/layer_groups.json"),
                    help="Path to layer_groups.json config file")
    ap.add_argument("--components-per-layer", type=int, default=2,
                    help="Number of PC components per layer in group (group PCA mode)")
    ap.add_argument("--temp-dir", type=Path, default=None,
                    help="Temporary directory for activation storage (group PCA mode)")
    ap.add_argument("--image-list", type=Path, default=None,
                    help="Text file containing list of image IDs to process (one per line)")

    args = ap.parse_args()

    run_convnext = not args.disable_convnext
    run_clip = not args.disable_clip
    run_vgg = args.enable_vgg and not args.disable_vgg

    if not run_convnext and not run_clip and not run_vgg:
        raise ValueError("All models are disabled; nothing to do.")

    # ConvNeXt requires a checkpoint
    if run_convnext and args.convnext_checkpoint is None:
        raise ValueError("--convnext-checkpoint is required when running ConvNeXt")

    # VGG requires a checkpoint
    if run_vgg and args.vgg_checkpoint is None:
        raise ValueError("--vgg-checkpoint is required when running VGG")

    if not args.image_dir.exists():
        raise ValueError(f"Image directory not found: {args.image_dir}")

    if args.use_group_pca and not args.layer_groups_config.exists():
        raise ValueError(f"Layer groups config not found: {args.layer_groups_config}")

    # Collect all image paths
    image_paths = sorted(args.image_dir.glob("*.jpg")) + sorted(args.image_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    # Filter by image list if provided
    if args.image_list:
        if not args.image_list.exists():
            raise ValueError(f"Image list file not found: {args.image_list}")

        # Load image IDs from file
        with open(args.image_list, 'r') as f:
            allowed_ids = set(line.strip() for line in f if line.strip())

        # Normalize IDs to 4-digit format
        normalized_allowed_ids = set(id.zfill(4) for id in allowed_ids)

        # Filter image paths
        filtered_paths = [
            path for path in image_paths
            if path.stem in normalized_allowed_ids or path.stem in allowed_ids
        ]

        print(f"Found {len(image_paths)} images in {args.image_dir}")
        print(f"Filtered to {len(filtered_paths)} images from {args.image_list}")
        image_paths = filtered_paths
    else:
        print(f"Found {len(image_paths)} images in {args.image_dir}")

    if args.use_group_pca:
        print(f"\n=== Using LayerGroup-based PCA ===")
        print(f"Config: {args.layer_groups_config}")
        print(f"Components per layer: {args.components_per_layer}")
    else:
        print(f"\n=== Using per-layer PCA ===")
        print(f"Max components per layer: {args.max_components}")

    image_id_list = [path.stem for path in image_paths]

    # Get device
    device = get_device()

    # Extract ConvNeXt features
    convnext_results: List[Dict[str, Any]] = []
    if run_convnext:
        print("\n" + "="*60)
        print("EXTRACTING CONVNEXT FEATURES")
        print("="*60)

        if args.use_group_pca:
            convnext_results = extract_convnext_activations_group_pca(
                args.convnext_checkpoint,
                image_paths,
                args.layer_groups_config,
                args.components_per_layer,
                device,
                args.batch_size,
                args.temp_dir,
            )
        else:
            convnext_results = extract_convnext_activations(
                args.convnext_checkpoint,
                image_paths,
                args.max_components,
                device,
                args.batch_size,
                args.temp_dir,
            )

    # Compute behavior ratings
    rating_means = compute_mean_behavior_ratings(args.behavior_dir)

    if run_convnext:
        if args.use_group_pca:
            # For group PCA, determine max components from largest group
            max_rating_components = 20  # Initial group has 10 layers * 2 = 20
        else:
            max_rating_components = args.max_components

        append_rating_layer(
            convnext_results,
            rating_means,
            max_rating_components,
            module_name="convnext.rating_mean",
            image_ids_override=image_id_list,
        )

    # Extract CLIP features
    clip_results: List[Dict[str, Any]] = []
    if run_clip:
        print("\n" + "="*60)
        print("EXTRACTING CLIP FEATURES")
        print("="*60)

        if args.use_group_pca:
            clip_results = extract_clip_activations_group_pca(
                image_paths,
                args.layer_groups_config,
                args.components_per_layer,
                device,
                args.batch_size,
                args.temp_dir,
            )
        else:
            clip_results = extract_clip_activations(
                image_paths,
                args.max_components,
                device,
                args.batch_size,
                args.temp_dir,
            )

    if run_clip:
        if args.use_group_pca:
            max_rating_components = 20
        else:
            max_rating_components = args.max_components

        append_rating_layer(
            clip_results,
            rating_means,
            max_rating_components,
            module_name="clip.rating_mean",
            image_ids_override=image_id_list,
        )

    # Extract VGG features
    vgg_results: List[Dict[str, Any]] = []
    if run_vgg:
        print("\n" + "="*60)
        print("EXTRACTING VGG FEATURES")
        print("="*60)

        if args.use_group_pca:
            vgg_results = extract_vgg_activations_group_pca(
                args.vgg_checkpoint,
                image_paths,
                args.layer_groups_config,
                args.components_per_layer,
                device,
                args.batch_size,
                args.temp_dir,
            )
        else:
            vgg_results = extract_vgg_activations(
                args.vgg_checkpoint,
                image_paths,
                args.max_components,
                device,
                args.batch_size,
            )

    if run_vgg:
        if args.use_group_pca:
            max_rating_components = 10  # VGG has max 5 layers per group * 2
        else:
            max_rating_components = args.max_components

        append_rating_layer(
            vgg_results,
            rating_means,
            max_rating_components,
            module_name="vgg.rating_mean",
            image_ids_override=image_id_list,
        )

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0

    if run_convnext and convnext_results:
        convnext_path = args.out_dir / "convnext_pcs.csv"

        # Determine max components for CSV columns
        if args.use_group_pca:
            # Find max components across all results
            max_components_csv = max(len(r["pc_scores"]) for r in convnext_results)
        else:
            max_components_csv = args.max_components

        save_results_to_csv(convnext_results, convnext_path, max_components_csv)
        n_images = len(set(r["image_id"] for r in convnext_results))
        n_modules = len(set(r["module_name"] for r in convnext_results))

        if args.use_group_pca:
            print(f"\nWrote {convnext_path}: {n_images} images × {n_modules} groups = {len(convnext_results)} rows")
        else:
            print(f"\nWrote {convnext_path}: {n_images} images × {n_modules} layers = {len(convnext_results)} rows")

        total_rows += len(convnext_results)
    elif run_convnext:
        print("Warning: No ConvNeXt results generated")

    if run_clip and clip_results:
        clip_path = args.out_dir / "clip_pcs.csv"

        # Determine max components for CSV columns
        if args.use_group_pca:
            max_components_csv = max(len(r["pc_scores"]) for r in clip_results)
        else:
            max_components_csv = args.max_components

        save_results_to_csv(clip_results, clip_path, max_components_csv)
        n_images = len(set(r["image_id"] for r in clip_results))
        n_modules = len(set(r["module_name"] for r in clip_results))

        if args.use_group_pca:
            print(f"\nWrote {clip_path}: {n_images} images × {n_modules} groups = {len(clip_results)} rows")
        else:
            print(f"\nWrote {clip_path}: {n_images} images × {n_modules} layers = {len(clip_results)} rows")

        total_rows += len(clip_results)
    elif run_clip:
        print("Warning: No CLIP results generated")

    if run_vgg and vgg_results:
        vgg_path = args.out_dir / "vgg_pcs.csv"

        # Determine max components for CSV columns
        if args.use_group_pca:
            max_components_csv = max(len(r["pc_scores"]) for r in vgg_results)
        else:
            max_components_csv = args.max_components

        save_results_to_csv(vgg_results, vgg_path, max_components_csv)
        n_images = len(set(r["image_id"] for r in vgg_results))
        n_modules = len(set(r["module_name"] for r in vgg_results))

        if args.use_group_pca:
            print(f"\nWrote {vgg_path}: {n_images} images × {n_modules} groups = {len(vgg_results)} rows")
        else:
            print(f"\nWrote {vgg_path}: {n_images} images × {n_modules} layers = {len(vgg_results)} rows")

        total_rows += len(vgg_results)
    elif run_vgg:
        print("Warning: No VGG results generated")

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows: {total_rows}")
    if args.use_group_pca:
        print(f"Mode: LayerGroup-based PCA ({args.components_per_layer} components per layer)")
    else:
        print(f"Mode: Per-layer PCA ({args.max_components} components per layer)")


if __name__ == "__main__":
    main()
