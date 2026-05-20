from pathlib import Path
import argparse
from collections import Counter, defaultdict
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_image_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image


def save_overlay(image, mask, save_path: Path, title: str = ""):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(mask > 0, alpha=0.4, cmap="jet")
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def find_manifest(root: Path):
    csv_path = root / "manifest.csv"
    json_path = root / "manifest.json"

    if csv_path.exists():
        return csv_path

    if json_path.exists():
        return json_path

    return None


def analyse_manifest(manifest_path: Path, out_tables: Path):
    print("\n[INFO] Analysing manifest...")

    if manifest_path.suffix.lower() == ".csv":
        df = pd.read_csv(manifest_path)
    else:
        df = pd.read_json(manifest_path)

    out_tables.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_tables / "brisc_manifest_loaded.csv", index=False)

    print("\nManifest columns:")
    print(df.columns.tolist())

    summary = {
        "total_manifest_rows": len(df)
    }

    if "task" in df.columns:
        task_counts = df["task"].value_counts().reset_index()
        task_counts.columns = ["task", "count"]
        task_counts.to_csv(out_tables / "brisc_task_distribution.csv", index=False)

    if "split" in df.columns:
        split_counts = df["split"].value_counts().reset_index()
        split_counts.columns = ["split", "count"]
        split_counts.to_csv(out_tables / "brisc_split_distribution.csv", index=False)

    if "tumor_label" in df.columns:
        class_counts = df.groupby(["split", "tumor_label"]).size().reset_index(name="count")
        class_counts.to_csv(out_tables / "brisc_class_distribution.csv", index=False)

    if "plane_label" in df.columns:
        plane_counts = df.groupby(["split", "plane_label"]).size().reset_index(name="count")
        plane_counts.to_csv(out_tables / "brisc_plane_distribution.csv", index=False)

    if "sequence" in df.columns:
        sequence_counts = df["sequence"].value_counts().reset_index()
        sequence_counts.columns = ["sequence", "count"]
        sequence_counts.to_csv(out_tables / "brisc_sequence_distribution.csv", index=False)

    if "width" in df.columns and "height" in df.columns:
        size_counts = df.groupby(["width", "height"]).size().reset_index(name="count")
        size_counts.to_csv(out_tables / "brisc_manifest_size_distribution.csv", index=False)

    pd.DataFrame([summary]).to_csv(out_tables / "brisc_manifest_summary.csv", index=False)

    return df


def analyse_classification_task(root: Path, out_tables: Path):
    print("\n[INFO] Analysing classification_task folder...")

    classification_root = root / "classification_task"

    if not classification_root.exists():
        print("[WARNING] classification_task folder not found.")
        return

    rows = []

    for class_dir in sorted(classification_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        image_paths = list(class_dir.glob("*.*"))

        for img_path in image_paths:
            image = read_image_gray(img_path)

            if image is None:
                rows.append({
                    "class_name": class_name,
                    "filename": img_path.name,
                    "status": "corrupt_image"
                })
                continue

            h, w = image.shape

            rows.append({
                "class_name": class_name,
                "filename": img_path.name,
                "status": "ok",
                "width": w,
                "height": h,
                "image_min": float(image.min()),
                "image_max": float(image.max()),
                "image_mean": float(image.mean()),
                "image_std": float(image.std())
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_tables / "brisc_classification_folder_report.csv", index=False)

    if len(df) > 0:
        class_counts = df.groupby(["class_name", "status"]).size().reset_index(name="count")
        class_counts.to_csv(out_tables / "brisc_classification_folder_class_counts.csv", index=False)

        size_counts = df.groupby(["width", "height"]).size().reset_index(name="count")
        size_counts.to_csv(out_tables / "brisc_classification_folder_size_counts.csv", index=False)

    print("[INFO] Classification folder analysis complete.")


def analyse_segmentation_task(root: Path, out_tables: Path, out_figures: Path, num_overlays: int):
    print("\n[INFO] Analysing segmentation_task folder...")

    image_dir = root / "segmentation_task" / "train" / "images"
    mask_dir = root / "segmentation_task" / "train" / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Segmentation image folder not found: {image_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Segmentation mask folder not found: {mask_dir}")

    image_paths = sorted(list(image_dir.glob("*.*")))
    mask_paths = sorted(list(mask_dir.glob("*.*")))

    print(f"Segmentation images found: {len(image_paths)}")
    print(f"Segmentation masks found: {len(mask_paths)}")

    mask_stems = {p.stem: p for p in mask_paths}

    rows = []
    global_mask_values = Counter()
    overlay_candidates = []

    for img_path in tqdm(image_paths, desc="Checking segmentation pairs"):
        expected_mask_path = mask_stems.get(img_path.stem, None)

        row = {
            "image_filename": img_path.name,
            "image_stem": img_path.stem,
            "mask_filename": expected_mask_path.name if expected_mask_path else None,
            "has_mask": expected_mask_path is not None
        }

        image = read_image_gray(img_path)

        if image is None:
            row["status"] = "corrupt_image"
            rows.append(row)
            continue

        ih, iw = image.shape

        row.update({
            "image_width": iw,
            "image_height": ih,
            "image_min": float(image.min()),
            "image_max": float(image.max()),
            "image_mean": float(image.mean()),
            "image_std": float(image.std())
        })

        if expected_mask_path is None:
            row["status"] = "missing_mask"
            rows.append(row)
            continue

        mask = read_image_gray(expected_mask_path)

        if mask is None:
            row["status"] = "corrupt_mask"
            rows.append(row)
            continue

        mh, mw = mask.shape
        unique_values = np.unique(mask).tolist()

        for value in unique_values:
            global_mask_values[int(value)] += 1

        tumour_pixels = int(np.sum(mask > 0))
        total_pixels = int(mask.size)
        tumour_ratio = tumour_pixels / total_pixels if total_pixels > 0 else 0.0

        row.update({
            "status": "ok",
            "mask_width": mw,
            "mask_height": mh,
            "dimension_match": bool(iw == mw and ih == mh),
            "unique_mask_values": str(unique_values),
            "is_binary_mask": set(unique_values).issubset({0, 1, 255}),
            "has_tumour_pixels": tumour_pixels > 0,
            "tumour_pixels": tumour_pixels,
            "total_pixels": total_pixels,
            "tumour_ratio": tumour_ratio
        })

        rows.append(row)

        if tumour_pixels > 0:
            overlay_candidates.append((img_path, expected_mask_path, tumour_ratio))

    df = pd.DataFrame(rows)
    df.to_csv(out_tables / "brisc_segmentation_pair_report.csv", index=False)

    mask_value_df = pd.DataFrame(
        [{"mask_value": k, "number_of_masks_containing_value": v} for k, v in sorted(global_mask_values.items())]
    )
    mask_value_df.to_csv(out_tables / "brisc_mask_value_report.csv", index=False)

    if len(df) > 0:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        status_counts.to_csv(out_tables / "brisc_segmentation_status_counts.csv", index=False)

        size_counts = df.groupby(["image_width", "image_height"]).size().reset_index(name="count")
        size_counts.to_csv(out_tables / "brisc_segmentation_image_size_counts.csv", index=False)

        tumour_summary = {
            "total_images": len(df),
            "ok_images": int((df["status"] == "ok").sum()),
            "images_with_tumour_pixels": int(df.get("has_tumour_pixels", pd.Series(dtype=bool)).sum()),
            "empty_masks": int((df.get("has_tumour_pixels", pd.Series(dtype=bool)) == False).sum()),
            "mean_tumour_ratio": float(df[df["status"] == "ok"]["tumour_ratio"].mean()),
            "median_tumour_ratio": float(df[df["status"] == "ok"]["tumour_ratio"].median()),
            "max_tumour_ratio": float(df[df["status"] == "ok"]["tumour_ratio"].max()),
            "min_tumour_ratio": float(df[df["status"] == "ok"]["tumour_ratio"].min())
        }

        pd.DataFrame([tumour_summary]).to_csv(out_tables / "brisc_tumour_pixel_summary.csv", index=False)

    print("\n[INFO] Saving sample overlays...")

    out_overlay_dir = out_figures / "brisc_sample_overlays"
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    overlay_candidates = sorted(overlay_candidates, key=lambda x: x[2], reverse=True)

    if len(overlay_candidates) > 0:
        selected = random.sample(
            overlay_candidates,
            min(num_overlays, len(overlay_candidates))
        )

        for idx, (img_path, mask_path, tumour_ratio) in enumerate(selected, start=1):
            image = read_image_gray(img_path)
            mask = read_image_gray(mask_path)

            save_overlay(
                image=image,
                mask=mask,
                save_path=out_overlay_dir / f"overlay_{idx:03d}_{img_path.stem}.png",
                title=f"{img_path.name} | tumour ratio: {tumour_ratio:.4f}"
            )

    print("[INFO] Segmentation analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="BRISC 2025 Dataset EDA Script")

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to BRISC2025 dataset root folder"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output folder for reports and figures"
    )

    parser.add_argument(
        "--num-overlays",
        type=int,
        default=30,
        help="Number of sample overlays to save"
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    out_tables = out_root / "tables"
    out_figures = out_root / "figures"

    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BRISC 2025 DATASET EDA")
    print("=" * 80)
    print(f"Dataset root: {root}")
    print(f"Output root: {out_root}")

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    manifest_path = find_manifest(root)

    if manifest_path:
        print(f"\n[INFO] Manifest found: {manifest_path}")
        analyse_manifest(manifest_path, out_tables)
    else:
        print("\n[WARNING] No manifest.csv or manifest.json found.")

    analyse_classification_task(root, out_tables)

    analyse_segmentation_task(
        root=root,
        out_tables=out_tables,
        out_figures=out_figures,
        num_overlays=args.num_overlays
    )

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)
    print(f"Reports saved in: {out_tables}")
    print(f"Figures saved in: {out_figures}")


if __name__ == "__main__":
    main()