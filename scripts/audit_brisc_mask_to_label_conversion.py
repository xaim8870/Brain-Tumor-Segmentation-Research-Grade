# scripts/audit_brisc_mask_to_label_conversion.py

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

RAW_IMG_DIR = Path(r"D:\Brain Tumor Segmentation\data\raw\BRICS\brisc2025\segmentation_task\train\images")
RAW_MASK_DIR = Path(r"D:\Brain Tumor Segmentation\data\raw\BRICS\brisc2025\segmentation_task\train\masks")

YOLO_IMG_DIR = Path(r"D:\Brain Tumor Segmentation\data\processed\brisc_yolo_seg_clean\images\train")
YOLO_LABEL_DIR = Path(r"D:\Brain Tumor Segmentation\data\processed\brisc_yolo_seg_clean\labels\train")

OUT_DIR = Path(r"D:\Brain Tumor Segmentation\results\dataset_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 128


def dice_score(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else (2 * inter) / denom


def iou_score(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def yolo_txt_to_mask(txt_path, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not txt_path.exists():
        return mask, 0

    lines = txt_path.read_text().strip().splitlines()
    polygon_count = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        coords = list(map(float, parts[1:]))
        points = []

        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * w))
            y = int(round(coords[i + 1] * h))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            points.append([x, y])

        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            polygon_count += 1

    return mask, polygon_count


rows = []

for img_path in sorted(YOLO_IMG_DIR.glob("*")):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    stem = img_path.stem

    raw_mask_path = RAW_MASK_DIR / f"{stem}.png"
    if not raw_mask_path.exists():
        raw_mask_path = RAW_MASK_DIR / f"{stem}.jpg"

    txt_path = YOLO_LABEL_DIR / f"{stem}.txt"

    raw_mask = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)

    if raw_mask is None:
        rows.append({
            "file": img_path.name,
            "status": "missing_or_corrupt_raw_mask"
        })
        continue

    raw_binary = (raw_mask >= THRESHOLD).astype(np.uint8) * 255
    yolo_mask, polygon_count = yolo_txt_to_mask(txt_path, raw_binary.shape)

    dice = dice_score(raw_binary, yolo_mask)
    iou = iou_score(raw_binary, yolo_mask)

    raw_area = int((raw_binary > 0).sum())
    yolo_area = int((yolo_mask > 0).sum())

    area_diff = abs(raw_area - yolo_area)
    area_diff_percent = (area_diff / raw_area * 100) if raw_area > 0 else 0

    rows.append({
        "file": img_path.name,
        "status": "ok",
        "polygon_count": polygon_count,
        "raw_area_pixels": raw_area,
        "yolo_area_pixels": yolo_area,
        "area_diff_pixels": area_diff,
        "area_diff_percent": area_diff_percent,
        "polygon_dice": dice,
        "polygon_iou": iou,
    })

df = pd.DataFrame(rows)

df.to_csv(OUT_DIR / "mask_to_yolo_label_audit.csv", index=False)

problem_df = df[
    (df["status"] != "ok") |
    (df["polygon_dice"] < 0.98) |
    (df["area_diff_percent"] > 5)
]

problem_df.to_csv(OUT_DIR / "problematic_mask_label_conversions.csv", index=False)

print("Audit saved to:", OUT_DIR)
print("\nSummary:")
print(df.describe())

print("\nProblematic samples:", len(problem_df))