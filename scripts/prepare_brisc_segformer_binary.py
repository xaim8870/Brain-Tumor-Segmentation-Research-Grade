from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_mask_binary(path: Path, threshold: int):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    if mask.ndim == 3:
        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = mask[:, :, 0]

    return (mask >= threshold).astype(np.uint8)


def get_image_path(row):
    for col in ["output_image_path", "image_path"]:
        if col in row and pd.notna(row[col]):
            p = Path(str(row[col]))
            if p.exists():
                return p
    raise FileNotFoundError("No valid image path found in output_image_path or image_path.")


def process_split(split_name, csv_path, out_root, imgsz, threshold):
    df = pd.read_csv(csv_path)

    img_out_dir = out_root / "images" / split_name
    mask_out_dir = out_root / "masks" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    stats = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        img_path = get_image_path(row)
        mask_path = Path(str(row["mask_path"]))

        img = read_image(img_path)
        mask = read_mask_binary(mask_path, threshold)

        img_resized = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)

        out_img_path = img_out_dir / f"{img_path.stem}.png"
        out_mask_path = mask_out_dir / f"{img_path.stem}.png"

        cv2.imwrite(str(out_img_path), img_resized)
        cv2.imwrite(str(out_mask_path), mask_resized)

        area_pixels = int(mask_resized.sum())
        area_percent = float(area_pixels / (imgsz * imgsz) * 100)

        rows.append({
            "image_path": str(out_img_path),
            "mask_path": str(out_mask_path),
            "tumour_label": row.get("tumour_label", "unknown"),
            "plane_label": row.get("plane_label", "unknown"),
            "class_id": int(row["class_id"]) if "class_id" in row and pd.notna(row["class_id"]) else -1,
            "original_image_path": str(img_path),
            "original_mask_path": str(mask_path),
        })

        stats.append({
            "split": split_name,
            "file": img_path.name,
            "tumour_label": row.get("tumour_label", "unknown"),
            "plane_label": row.get("plane_label", "unknown"),
            "mask_area_pixels": area_pixels,
            "mask_area_percent": area_percent,
            "mask_unique_values": sorted(np.unique(mask_resized).tolist()),
        })

    out_df = pd.DataFrame(rows)
    stats_df = pd.DataFrame(stats)

    out_csv = out_root / f"segformer_{split_name}.csv"
    out_df.to_csv(out_csv, index=False)

    return out_df, stats_df


def save_overlay_samples(out_root, split_csv, split_name, max_samples=30):
    df = pd.read_csv(split_csv)
    overlay_dir = out_root / "overlay_samples" / split_name
    overlay_dir.mkdir(parents=True, exist_ok=True)

    sample_df = df.sample(min(max_samples, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        img = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        overlay = img.copy()
        overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)

        out_path = overlay_dir / Path(row["image_path"]).name
        cv2.imwrite(str(out_path), overlay)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_stats = process_split(
        "train",
        args.train_csv,
        out_root,
        args.imgsz,
        args.threshold,
    )

    val_df, val_stats = process_split(
        "val",
        args.val_csv,
        out_root,
        args.imgsz,
        args.threshold,
    )

    all_stats = pd.concat([train_stats, val_stats], ignore_index=True)
    all_stats.to_csv(reports_dir / "mask_area_summary.csv", index=False)

    pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
    ]).groupby(["split", "tumour_label"]).size().reset_index(name="count").to_csv(
        reports_dir / "class_distribution.csv",
        index=False,
    )

    all_stats.groupby(["split", "tumour_label"])[
        ["mask_area_pixels", "mask_area_percent"]
    ].describe().to_csv(
        reports_dir / "class_mask_area_numeric_summary.csv"
    )

    save_overlay_samples(out_root, out_root / "segformer_train.csv", "train")
    save_overlay_samples(out_root, out_root / "segformer_val.csv", "val")

    print("\nSegFormer binary preprocessing complete.")
    print("Output:", out_root)
    print("Train CSV:", out_root / "segformer_train.csv")
    print("Val CSV:", out_root / "segformer_val.csv")
    print("Reports:", reports_dir)


if __name__ == "__main__":
    main()