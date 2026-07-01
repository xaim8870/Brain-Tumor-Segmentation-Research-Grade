import os
import cv2
import numpy as np
from tqdm import tqdm

# Change these paths according to your dataset
IMAGE_DIR = r"D:\Brain Tumor Segmentation\data\processed\brisc_segformer_binary\val\images"
MASK_DIR  = r"D:\Brain Tumor Segmentation\data\processed\brisc_segformer_binary\val\masks"
OUT_DIR   = r"D:\Brain Tumor Segmentation\results\overlay_images"

os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.45  # overlay transparency

for img_name in tqdm(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {img_name}")
        continue

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    # Resize mask if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Binary mask
    mask_bin = mask > 127

    # Create red overlay
    overlay = image.copy()
    overlay[mask_bin] = (0, 0, 255)  # Red in OpenCV BGR

    # Blend original image and overlay
    blended = cv2.addWeighted(image, 1 - ALPHA, overlay, ALPHA, 0)

    # Optional: add tumor boundary
    contours, _ = cv2.findContours(
        mask_bin.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (0, 255, 255), 2)  # Yellow boundary

    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, blended)

print("Overlay images saved at:", OUT_DIR)