import os
import shutil
import random
from glob import glob

# --- CONFIG ---
base_dir = "dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# fractions (must sum to 1.0)
splits_frac = {
    "train": 0.8,
    "val":   0.1,
    "test":  0.1
}

# Optional: set a seed for reproducible shuffling (remove or change for different random split)
random_seed = 42
# ----------------

# Create output folders
for split in splits_frac:
    os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "labels"), exist_ok=True)

# Collect image files (accept .jpg, .jpeg, .png, case-insensitive)
valid_exts = {".jpg", ".jpeg", ".png"}
image_files = [
    f for f in os.listdir(images_dir)
    if os.path.isfile(os.path.join(images_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
]

image_files.sort()
if random_seed is not None:
    random.seed(random_seed)
random.shuffle(image_files)

# Compute split indices
total = len(image_files)
if total == 0:
    raise SystemExit(f"No images found in {images_dir} with extensions {valid_exts}")

train_end = int(splits_frac["train"] * total)
val_end = train_end + int(splits_frac["val"] * total)

splits = {
    "train": image_files[:train_end],
    "val":   image_files[train_end:val_end],
    "test":  image_files[val_end:]
}

def find_label_for_base(base_name):
    """Find any file in labels_dir that starts with base_name (exact base) and return its path, or None."""
    # exact common extensions first
    candidates = [os.path.join(labels_dir, base_name + ext) for ext in (".png", ".txt", ".json")]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: find any file with the same base name (any extension)
    pattern = os.path.join(labels_dir, base_name + ".*")
    matches = glob(pattern)
    return matches[0] if matches else None

# Copy files
for split, files in splits.items():
    for img_file in files:
        name, _ = os.path.splitext(img_file)
        src_img = os.path.join(images_dir, img_file)
        src_lbl = find_label_for_base(name)

        dst_img = os.path.join(base_dir, split, "images", img_file)
        if src_lbl:
            dst_lbl = os.path.join(base_dir, split, "labels", os.path.basename(src_lbl))
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # No label found — copy image anyway or decide to skip
            shutil.copy2(src_img, dst_img)
            print(f"Warning: Missing label for {img_file} (expected base: {name})")
