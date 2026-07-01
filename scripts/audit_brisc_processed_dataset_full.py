from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_MASK_DIR = Path(r"D:\Brain Tumor Segmentation\data\raw\BRICS\brisc2025\segmentation_task\train\masks")

PROCESSED_IMG_DIR = Path(r"D:\Brain Tumor Segmentation\data\processed\brisc_yolo_seg_clean\images\train")
PROCESSED_LABEL_DIR = Path(r"D:\Brain Tumor Segmentation\data\processed\brisc_yolo_seg_clean\labels\train")

SPLIT_CSV = Path(r"D:\Brain Tumor Segmentation\data\splits_clean\brisc_yolo_train.csv")

OUT_DIR = Path(r"D:\Brain Tumor Segmentation\results\dataset_audit_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 128

CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary"
}


def dice_score(a, b):
    a = a > 0
    b = b > 0
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2 * np.logical_and(a, b).sum() / denom


def iou_score(a, b):
    a = a > 0
    b = b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return np.logical_and(a, b).sum() / union


def yolo_to_mask(label_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon_count = 0
    bbox_areas = []

    if not label_path.exists():
        return mask, polygon_count, bbox_areas

    lines = label_path.read_text().strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        coords = list(map(float, parts[1:]))
        pts = []

        xs, ys = [], []

        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * w))
            y = int(round(coords[i + 1] * h))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])
            xs.append(x)
            ys.append(y)

        if len(pts) >= 3:
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            polygon_count += 1

            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            bbox_areas.append(bw * bh)

    return mask, polygon_count, bbox_areas


def find_raw_mask(stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        p = RAW_MASK_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


rows = []

df = pd.read_csv(SPLIT_CSV)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Auditing dataset"):
    img_path = Path(row["output_image_path"]) if "output_image_path" in row else None

    if img_path is None or not img_path.exists():
        img_path = PROCESSED_IMG_DIR / Path(row["image_path"]).name

    stem = img_path.stem
    label_path = PROCESSED_LABEL_DIR / f"{stem}.txt"
    raw_mask_path = find_raw_mask(stem)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        rows.append({
            "file": stem,
            "status": "missing_image"
        })
        continue

    h, w = img.shape[:2]

    raw_mask = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE) if raw_mask_path else None

    if raw_mask is None:
        rows.append({
            "file": stem,
            "status": "missing_raw_mask",
            "height": h,
            "width": w,
        })
        continue

    raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    raw_binary = (raw_mask >= THRESHOLD).astype(np.uint8) * 255

    yolo_mask, polygon_count, bbox_areas = yolo_to_mask(label_path, h, w)

    raw_area = int((raw_binary > 0).sum())
    yolo_area = int((yolo_mask > 0).sum())

    img_area = h * w
    tumour_area_percent = (raw_area / img_area) * 100 if img_area > 0 else 0

    bbox_area = max(bbox_areas) if bbox_areas else 0
    bbox_area_percent = (bbox_area / img_area) * 100 if img_area > 0 else 0

    poly_dice = dice_score(raw_binary, yolo_mask)
    poly_iou = iou_score(raw_binary, yolo_mask)

    class_id = int(row["class_id"]) if "class_id" in row else -1
    tumour_label = row.get("tumour_label", CLASS_NAMES.get(class_id, "unknown"))
    plane_label = row.get("plane_label", "unknown")

    rows.append({
        "file": stem,
        "status": "ok",
        "tumour_label": tumour_label,
        "class_id": class_id,
        "plane_label": plane_label,
        "height": h,
        "width": w,
        "image_area": img_area,
        "raw_mask_area": raw_area,
        "yolo_mask_area": yolo_area,
        "tumour_area_percent": tumour_area_percent,
        "bbox_area": bbox_area,
        "bbox_area_percent": bbox_area_percent,
        "polygon_count": polygon_count,
        "polygon_dice": poly_dice,
        "polygon_iou": poly_iou,
        "area_diff_pixels": abs(raw_area - yolo_area),
        "area_diff_percent": abs(raw_area - yolo_area) / raw_area * 100 if raw_area > 0 else 0,
        "label_exists": label_path.exists(),
    })


audit_df = pd.DataFrame(rows)
audit_df.to_csv(OUT_DIR / "full_processed_dataset_audit.csv", index=False)

ok_df = audit_df[audit_df["status"] == "ok"].copy()

# Main summaries
ok_df.groupby("tumour_label").size().reset_index(name="count").to_csv(
    OUT_DIR / "class_distribution.csv", index=False
)

ok_df.groupby("plane_label").size().reset_index(name="count").to_csv(
    OUT_DIR / "plane_distribution.csv", index=False
)

ok_df.groupby(["tumour_label", "plane_label"]).size().reset_index(name="count").to_csv(
    OUT_DIR / "class_plane_distribution.csv", index=False
)

ok_df.groupby("tumour_label")[
    [
        "raw_mask_area",
        "tumour_area_percent",
        "bbox_area_percent",
        "polygon_dice",
        "polygon_iou",
        "area_diff_percent",
        "polygon_count",
    ]
].describe().to_csv(
    OUT_DIR / "class_wise_numeric_summary.csv"
)

ok_df.groupby(["tumour_label", "plane_label"])[
    [
        "raw_mask_area",
        "tumour_area_percent",
        "bbox_area_percent",
        "polygon_dice",
        "polygon_iou",
        "area_diff_percent",
    ]
].describe().to_csv(
    OUT_DIR / "class_plane_numeric_summary.csv"
)

# Problematic samples
problem_df = ok_df[
    (ok_df["polygon_dice"] < 0.98) |
    (ok_df["polygon_iou"] < 0.96) |
    (ok_df["area_diff_percent"] > 5) |
    (ok_df["tumour_area_percent"] < 0.15)
].copy()

problem_df.to_csv(OUT_DIR / "problematic_samples.csv", index=False)

# Worst samples per class
for cls in ok_df["tumour_label"].unique():
    cls_df = ok_df[ok_df["tumour_label"] == cls].copy()

    cls_df.sort_values("polygon_dice").head(50).to_csv(
        OUT_DIR / f"worst_{cls}_polygon_conversion.csv", index=False
    )

    cls_df.sort_values("tumour_area_percent").head(50).to_csv(
        OUT_DIR / f"smallest_{cls}_tumours.csv", index=False
    )

    cls_df.sort_values("bbox_area_percent").head(50).to_csv(
        OUT_DIR / f"smallest_{cls}_bbox.csv", index=False
    )

# Size bins
bins = [0, 0.15, 0.5, 1, 2, 5, 10, 100]
labels = [
    "tiny_<0.15%",
    "very_small_0.15-0.5%",
    "small_0.5-1%",
    "medium_1-2%",
    "large_2-5%",
    "very_large_5-10%",
    "huge_>10%",
]

ok_df["tumour_size_bin"] = pd.cut(
    ok_df["tumour_area_percent"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

ok_df.groupby(["tumour_label", "tumour_size_bin"]).size().reset_index(name="count").to_csv(
    OUT_DIR / "class_tumour_size_bins.csv", index=False
)

print("\nAudit complete.")
print("Saved to:", OUT_DIR)

print("\nClass distribution:")
print(ok_df.groupby("tumour_label").size())

print("\nPlane distribution:")
print(ok_df.groupby("plane_label").size())

print("\nClass × plane distribution:")
print(ok_df.groupby(["tumour_label", "plane_label"]).size())

print("\nMean by class:")
print(
    ok_df.groupby("tumour_label")[
        ["tumour_area_percent", "bbox_area_percent", "polygon_dice", "polygon_iou", "area_diff_percent"]
    ].mean()
)

print("\nProblematic samples:", len(problem_df))
print(problem_df.groupby("tumour_label").size())