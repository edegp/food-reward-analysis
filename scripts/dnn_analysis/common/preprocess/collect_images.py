#!/usr/bin/env python3
"""
Collect all image IDs used across all subjects in the experiment.

This script scans rating_data CSV files for all subjects to determine:
1. Which images were actually presented
2. How many unique images were used
3. Image presentation frequency across subjects
"""
import pandas as pd
from pathlib import Path
from collections import Counter

# Setup paths
behavior_dir = Path('/Users/yuhiaoki/dev/hit/food-brain/Food_Behavior')
subjects = [f'sub-{i:03d}' for i in range(1, 21)]  # sub-001 to sub-020

# Collect all image IDs
all_image_ids = []
subject_image_counts = {}

print("Collecting image IDs from all subjects...")
print("="*70)

for sub in subjects:
    sub_dir = behavior_dir / sub
    if not sub_dir.exists():
        print(f"  {sub}: Directory not found, skipping")
        continue

    # Find all rating_data CSV files
    rating_files = sorted(sub_dir.glob('rating_data*.csv'))

    if not rating_files:
        print(f"  {sub}: No rating files found")
        continue

    sub_images = []
    for rating_file in rating_files:
        df = pd.read_csv(rating_file)
        if 'Image Name' in df.columns:
            images = df['Image Name'].astype(str).tolist()
            sub_images.extend(images)

    subject_image_counts[sub] = len(sub_images)
    all_image_ids.extend(sub_images)
    print(f"  {sub}: {len(sub_images)} trials")

print()
print("="*70)
print("Summary Statistics")
print("="*70)

# Unique images
unique_images = sorted(set(all_image_ids))
print(f"Total trials across all subjects: {len(all_image_ids)}")
print(f"Unique images used: {len(unique_images)}")
print()

# Image frequency
image_freq = Counter(all_image_ids)
print("Image presentation frequency:")
freq_dist = Counter(image_freq.values())
for count in sorted(freq_dist.keys()):
    print(f"  Presented {count} times: {freq_dist[count]} images")

print()
print("Most frequently presented images:")
for img_id, count in image_freq.most_common(10):
    print(f"  {img_id}: {count} times")

print()
print("="*70)

# Save unique image list
output_file = Path('/Users/yuhiaoki/dev/hit/food-brain/data_images/used_image_ids.txt')
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    for img_id in unique_images:
        f.write(f"{img_id}\n")

print(f"Saved unique image IDs to: {output_file}")
print(f"Total: {len(unique_images)} images")
print("="*70)
