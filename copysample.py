import os
import shutil
import random

# === ✅ CONFIGURE THIS PATH ===
SOURCE_DIR = "/Users/cavins/Desktop/project/leaf-disease-detector/leaf_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"  # <-- Change to your actual dataset path
DEST_DIR = "samples"       # Output folder
NUM_IMAGES = 200           # Number of images to sample

# === Find all image files recursively ===
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
all_images = []

for root, _, files in os.walk(SOURCE_DIR):
    for f in files:
        if f.lower().endswith(image_extensions):
            all_images.append(os.path.join(root, f))

# === Check availability ===
if len(all_images) < NUM_IMAGES:
    raise ValueError(f"Only {len(all_images)} images found, but you requested {NUM_IMAGES}.")

# === Sample & copy ===
os.makedirs(DEST_DIR, exist_ok=True)
selected_images = random.sample(all_images, NUM_IMAGES)

for src_path in selected_images:
    dest_path = os.path.join(DEST_DIR, os.path.basename(src_path))
    shutil.copy(src_path, dest_path)

print(f"✅ Copied {NUM_IMAGES} images to '{DEST_DIR}' successfully.")
