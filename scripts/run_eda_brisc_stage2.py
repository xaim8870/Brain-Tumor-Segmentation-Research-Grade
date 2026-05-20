from pathlib import Path
import argparse
import re
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


TUMOUR_CODE_TO_LABEL = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "nt": "no_tumor",
}

PLANE_CODE_TO_LABEL = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagittal",
}


def parse_brisc_filename(filename: str):
    """
    Expected filename:
    brisc2025_train_00001_gl_ax_t1.jpg
    brisc2025_test_00010_me_co_t1.png
    """

    pattern = r"brisc2025_(train|test)_(\d+)_(gl|me|pi|nt)_(ax|co|sa)_(t1)"
    match = re.search(pattern, filename.lower())

    if match is None:
        return {
            "split": "unknown",
            "index": None,
            "tumour_code": "unknown",
            "tumour_label": "unknown",
            "plane_code": "unknown",
            "plane_label": "unknown",
            "sequence": "unknown",
        }

    split, index, tumour_code, plane_code, sequence = match.groups()

    return {
        "split": split,
        "index": int(index),
        "tumour_code": tumour_code,
        "tumour_label": TUMOUR_CODE_TO_LABEL.get(tumour_code, "unknown"),
        "plane_code": plane_code,
        "plane_label": PLANE_CODE_TO_LABEL.get(plane_code, "unknown"),
        "sequence": sequence.upper(),
    }


def read_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def find_existing_segmentation_root(root: Path):
    """
    Tries to find the folder that contains the segmentation data.

    Supported examples:

    brisc2025/
    └── segmentation_task/

    brisc2025/
    └── segmentation_mask/

    Or user can directly pass:
    segmentation_task/
    """

    candidates = [
        root / "segmentation_task",
        root / "segmentation_mask",
        root / "segmentation_masks",
        root / "segmentation",
        root,
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        has_split_structure = any(
            (candidate / split / "images").exists()
            and (candidate / split / "masks").exists()
            for split in ["train", "test", "val", "valid", "validation"]
        )

        has_flat_structure = (
            (candidate / "images").exists()
            and (candidate / "masks").exists()
        )

        has_images_masks_split_structure = any(
            (candidate / "images" / split).exists()
            and (candidate / "masks" / split).exists()
            for split in ["train", "test", "val", "valid", "validation"]
        )

        if has_split_structure or has_flat_structure or has_images_masks_split_structure:
            return candidate

    raise FileNotFoundError(
        "Could not find a valid segmentation folder. Expected one of these structures:\n"
        "1) root/segmentation_task/train/images and root/segmentation_task/train/masks\n"
        "2) root/segmentation_task/test/images and root/segmentation_task/test/masks\n"
        "3) root/segmentation_task/images and root/segmentation_task/masks\n"
        "4) root/segmentation_task/images/train and root/segmentation_task/masks/train"
    )


def find_segmentation_splits(root: Path):
    """
    Returns a dictionary like:

    {
        "train": {"image_dir": ..., "mask_dir": ...},
        "test": {"image_dir": ..., "mask_dir": ...}
    }

    Supported structures:

    Structure A:
    segmentation_task/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

    Structure B:
    segmentation_task/
    ├── images/
    └── masks/

    Structure C:
    segmentation_task/
    ├── images/
    │   ├── train/
    │   └── test/
    └── masks/
        ├── train/
        └── test/
    """

    segmentation_root = find_existing_segmentation_root(root)

    split_dirs = {}

    # Structure A: segmentation_task/train/images, segmentation_task/train/masks
    for split in ["train", "test", "val", "valid", "validation"]:
        image_dir = segmentation_root / split / "images"
        mask_dir = segmentation_root / split / "masks"

        if image_dir.exists() and mask_dir.exists():
            clean_split = "val" if split in ["valid", "validation"] else split
            split_dirs[clean_split] = {
                "image_dir": image_dir,
                "mask_dir": mask_dir,
            }

    if split_dirs:
        return segmentation_root, split_dirs

    # Structure C: segmentation_task/images/train, segmentation_task/masks/train
    for split in ["train", "test", "val", "valid", "validation"]:
        image_dir = segmentation_root / "images" / split
        mask_dir = segmentation_root / "masks" / split

        if image_dir.exists() and mask_dir.exists():
            clean_split = "val" if split in ["valid", "validation"] else split
            split_dirs[clean_split] = {
                "image_dir": image_dir,
                "mask_dir": mask_dir,
            }

    if split_dirs:
        return segmentation_root, split_dirs

    # Structure B: segmentation_task/images, segmentation_task/masks
    image_dir = segmentation_root / "images"
    mask_dir = segmentation_root / "masks"

    if image_dir.exists() and mask_dir.exists():
        split_dirs["all"] = {
            "image_dir": image_dir,
            "mask_dir": mask_dir,
        }
        return segmentation_root, split_dirs

    raise FileNotFoundError(
        f"No valid images/masks folders found inside: {segmentation_root}"
    )


def collect_files_recursively(folder: Path, allowed_suffixes):
    allowed_suffixes = {suffix.lower() for suffix in allowed_suffixes}
    files = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            files.append(path)

    return sorted(files)


def safe_savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bar_from_series(series, title, xlabel, ylabel, save_path: Path, rotate_x=False):
    plt.figure(figsize=(8, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0)

    safe_savefig(save_path)


def plot_image_size_distribution(df: pd.DataFrame, save_path: Path, top_n: int = 20):
    if df.empty:
        return

    size_counts = (
        df.groupby(["image_width", "image_height"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    size_counts["size"] = (
        size_counts["image_width"].astype(str)
        + "x"
        + size_counts["image_height"].astype(str)
    )

    top_sizes = size_counts.head(top_n)

    plt.figure(figsize=(12, 5))
    plt.bar(top_sizes["size"], top_sizes["count"])
    plt.title(f"Top {top_n} Image Size Distribution")
    plt.xlabel("Image Size")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")

    safe_savefig(save_path)


def plot_tumour_ratio_histogram(df: pd.DataFrame, save_path: Path):
    if df.empty:
        return

    values = df["tumour_ratio_threshold128"].dropna().values

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40)
    plt.title("Tumour Pixel Ratio Distribution")
    plt.xlabel("Tumour Pixel Ratio")
    plt.ylabel("Number of Images")

    safe_savefig(save_path)


def plot_boxplot_by_column(
    df: pd.DataFrame,
    column: str,
    value_column: str,
    title: str,
    save_path: Path,
):
    if df.empty:
        return

    groups = []
    labels = []

    for label, group in df.groupby(column):
        values = group[value_column].dropna().values
        if len(values) > 0:
            groups.append(values)
            labels.append(label)

    if not groups:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(groups, labels=labels)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel(value_column)
    plt.xticks(rotation=20, ha="right")

    safe_savefig(save_path)


def plot_mask_pixel_values(mask_value_counter: Counter, save_path: Path):
    if not mask_value_counter:
        return

    values = sorted(mask_value_counter.keys())
    counts = [mask_value_counter[v] for v in values]

    plt.figure(figsize=(12, 5))
    plt.bar(values, counts)
    plt.title("Raw Mask Pixel Value Distribution")
    plt.xlabel("Raw Mask Pixel Value")
    plt.ylabel("Total Pixel Count")
    plt.yscale("log")

    safe_savefig(save_path)


def save_summary_tables(full_df: pd.DataFrame, ok_df: pd.DataFrame, mask_value_counter: Counter, out_tables: Path):
    out_tables.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(out_tables / "brisc_stage2_full_report.csv", index=False)
    ok_df.to_csv(out_tables / "brisc_stage2_threshold128_report.csv", index=False)

    clean_summary = {
        "total_segmentation_files_checked": int(len(full_df)),
        "readable_image_mask_pairs": int(len(ok_df)),
        "missing_masks": int((full_df["status"] == "missing_mask").sum()) if "status" in full_df.columns else 0,
        "corrupt_images": int((full_df["status"] == "corrupt_image").sum()) if "status" in full_df.columns else 0,
        "corrupt_masks": int((full_df["status"] == "corrupt_mask").sum()) if "status" in full_df.columns else 0,
        "dimension_mismatch_count": int((ok_df["dimension_match"] == False).sum()) if not ok_df.empty else 0,
        "images_with_tumour_pixels_threshold128": int((ok_df["has_tumour_threshold128"] == True).sum()) if not ok_df.empty else 0,
        "empty_masks_threshold128": int((ok_df["has_tumour_threshold128"] == False).sum()) if not ok_df.empty else 0,
        "mean_tumour_ratio_threshold128": float(ok_df["tumour_ratio_threshold128"].mean()) if not ok_df.empty else 0.0,
        "median_tumour_ratio_threshold128": float(ok_df["tumour_ratio_threshold128"].median()) if not ok_df.empty else 0.0,
        "min_tumour_ratio_threshold128": float(ok_df["tumour_ratio_threshold128"].min()) if not ok_df.empty else 0.0,
        "max_tumour_ratio_threshold128": float(ok_df["tumour_ratio_threshold128"].max()) if not ok_df.empty else 0.0,
        "mean_image_intensity": float(ok_df["image_mean"].mean()) if not ok_df.empty else 0.0,
        "mean_image_std": float(ok_df["image_std"].mean()) if not ok_df.empty else 0.0,
    }

    pd.DataFrame([clean_summary]).to_csv(
        out_tables / "brisc_stage2_clean_summary.csv",
        index=False,
    )

    if ok_df.empty:
        return

    distribution_by_class = (
        ok_df.groupby(["split", "tumour_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "tumour_label"])
    )
    distribution_by_class.to_csv(
        out_tables / "brisc_stage2_distribution_by_class.csv",
        index=False,
    )

    distribution_by_plane = (
        ok_df.groupby(["split", "plane_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "plane_label"])
    )
    distribution_by_plane.to_csv(
        out_tables / "brisc_stage2_distribution_by_plane.csv",
        index=False,
    )

    tumour_ratio_by_class = (
        ok_df.groupby(["split", "tumour_label"])["tumour_ratio_threshold128"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    tumour_ratio_by_class.to_csv(
        out_tables / "brisc_stage2_tumour_ratio_by_class.csv",
        index=False,
    )

    tumour_ratio_by_plane = (
        ok_df.groupby(["split", "plane_label"])["tumour_ratio_threshold128"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    tumour_ratio_by_plane.to_csv(
        out_tables / "brisc_stage2_tumour_ratio_by_plane.csv",
        index=False,
    )

    image_size_distribution = (
        ok_df.groupby(["image_width", "image_height"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    image_size_distribution.to_csv(
        out_tables / "brisc_stage2_image_size_distribution.csv",
        index=False,
    )

    split_distribution = (
        ok_df.groupby(["split"])
        .size()
        .reset_index(name="count")
        .sort_values("split")
    )
    split_distribution.to_csv(
        out_tables / "brisc_stage2_split_distribution.csv",
        index=False,
    )

    mask_value_df = pd.DataFrame(
        [
            {
                "raw_mask_pixel_value": value,
                "total_pixel_count": count,
            }
            for value, count in sorted(mask_value_counter.items())
        ]
    )
    mask_value_df.to_csv(
        out_tables / "brisc_stage2_raw_mask_pixel_value_counts.csv",
        index=False,
    )


def save_graphs(ok_df: pd.DataFrame, mask_value_counter: Counter, out_figures: Path):
    out_figures.mkdir(parents=True, exist_ok=True)

    if ok_df.empty:
        print("[WARNING] No valid image-mask pairs found. Graphs were not created.")
        return

    class_series = ok_df["tumour_label"].value_counts().sort_index()
    plane_series = ok_df["plane_label"].value_counts().sort_index()
    split_series = ok_df["split"].value_counts().sort_index()

    plot_bar_from_series(
        class_series,
        title="BRISC Segmentation Class Distribution",
        xlabel="Tumour Class",
        ylabel="Number of Images",
        save_path=out_figures / "brisc_stage2_class_distribution.png",
        rotate_x=True,
    )

    plot_bar_from_series(
        plane_series,
        title="BRISC Segmentation Plane Distribution",
        xlabel="MRI Plane",
        ylabel="Number of Images",
        save_path=out_figures / "brisc_stage2_plane_distribution.png",
    )

    plot_bar_from_series(
        split_series,
        title="BRISC Segmentation Split Distribution",
        xlabel="Split",
        ylabel="Number of Images",
        save_path=out_figures / "brisc_stage2_split_distribution.png",
    )

    plot_image_size_distribution(
        df=ok_df,
        save_path=out_figures / "brisc_stage2_image_size_distribution.png",
        top_n=20,
    )

    plot_tumour_ratio_histogram(
        df=ok_df,
        save_path=out_figures / "brisc_stage2_tumour_ratio_histogram.png",
    )

    plot_boxplot_by_column(
        df=ok_df,
        column="tumour_label",
        value_column="tumour_ratio_threshold128",
        title="Tumour Pixel Ratio by Tumour Class",
        save_path=out_figures / "brisc_stage2_tumour_ratio_by_class.png",
    )

    plot_boxplot_by_column(
        df=ok_df,
        column="plane_label",
        value_column="tumour_ratio_threshold128",
        title="Tumour Pixel Ratio by MRI Plane",
        save_path=out_figures / "brisc_stage2_tumour_ratio_by_plane.png",
    )

    plot_mask_pixel_values(
        mask_value_counter=mask_value_counter,
        save_path=out_figures / "brisc_stage2_raw_mask_pixel_values.png",
    )


def run_stage2_analysis(root: Path, out_root: Path):
    segmentation_root, split_dirs = find_segmentation_splits(root)

    out_tables = out_root / "tables"
    out_figures = out_root / "figures"

    print("=" * 80)
    print("BRISC STAGE 2 EDA: GRAPHS + THRESHOLD-128 MASK ANALYSIS")
    print("=" * 80)
    print(f"Dataset root: {root}")
    print(f"Segmentation root detected: {segmentation_root}")
    print(f"Output root: {out_root}")
    print("=" * 80)

    rows = []
    raw_mask_value_counter = Counter()

    for folder_split, paths in split_dirs.items():
        image_dir = paths["image_dir"]
        mask_dir = paths["mask_dir"]

        image_paths = collect_files_recursively(
            image_dir,
            allowed_suffixes=[".jpg", ".jpeg", ".png"],
        )

        mask_paths = collect_files_recursively(
            mask_dir,
            allowed_suffixes=[".png", ".jpg", ".jpeg"],
        )

        mask_by_stem = {p.stem: p for p in mask_paths}

        print(f"\n[INFO] Processing split: {folder_split}")
        print(f"Image dir: {image_dir}")
        print(f"Mask dir: {mask_dir}")
        print(f"Images found: {len(image_paths)}")
        print(f"Masks found: {len(mask_paths)}")

        for image_path in tqdm(image_paths, desc=f"Stage 2 BRISC analysis [{folder_split}]"):
            parsed = parse_brisc_filename(image_path.name)

            row = {
                "folder_split": folder_split,
                "image_filename": image_path.name,
                "image_path": str(image_path),
                **parsed,
            }

            if row["split"] == "unknown":
                row["split"] = folder_split

            mask_path = mask_by_stem.get(image_path.stem)

            row["mask_filename"] = mask_path.name if mask_path else None
            row["mask_path"] = str(mask_path) if mask_path else None
            row["has_mask"] = mask_path is not None

            image = read_gray(image_path)

            if image is None:
                row["status"] = "corrupt_image"
                rows.append(row)
                continue

            image_height, image_width = image.shape

            row["image_width"] = image_width
            row["image_height"] = image_height
            row["image_min"] = float(image.min())
            row["image_max"] = float(image.max())
            row["image_mean"] = float(image.mean())
            row["image_std"] = float(image.std())

            if mask_path is None:
                row["status"] = "missing_mask"
                rows.append(row)
                continue

            mask = read_gray(mask_path)

            if mask is None:
                row["status"] = "corrupt_mask"
                rows.append(row)
                continue

            mask_height, mask_width = mask.shape

            row["mask_width"] = mask_width
            row["mask_height"] = mask_height
            row["dimension_match"] = bool(
                image_width == mask_width and image_height == mask_height
            )

            unique_values = np.unique(mask)
            row["raw_unique_mask_values"] = str(unique_values.tolist())
            row["raw_unique_mask_value_count"] = int(len(unique_values))

            values, counts = np.unique(mask, return_counts=True)
            for value, count in zip(values, counts):
                raw_mask_value_counter[int(value)] += int(count)

            # Correct binary interpretation for anti-aliased binary masks.
            # Values >= 128 are treated as tumour; values < 128 are background.
            mask_binary = mask >= 128

            tumour_pixels = int(mask_binary.sum())
            total_pixels = int(mask_binary.size)
            background_pixels = int(total_pixels - tumour_pixels)

            row["threshold_used"] = 128
            row["tumour_pixels_threshold128"] = tumour_pixels
            row["background_pixels_threshold128"] = background_pixels
            row["total_pixels"] = total_pixels
            row["tumour_ratio_threshold128"] = (
                tumour_pixels / total_pixels if total_pixels > 0 else 0.0
            )
            row["has_tumour_threshold128"] = tumour_pixels > 0
            row["status"] = "ok"

            rows.append(row)

    full_df = pd.DataFrame(rows)

    if full_df.empty:
        raise RuntimeError("No image files were found. Check the dataset root path.")

    ok_df = full_df[full_df["status"] == "ok"].copy()

    save_summary_tables(
        full_df=full_df,
        ok_df=ok_df,
        mask_value_counter=raw_mask_value_counter,
        out_tables=out_tables,
    )

    save_graphs(
        ok_df=ok_df,
        mask_value_counter=raw_mask_value_counter,
        out_figures=out_figures,
    )

    print("\n" + "=" * 80)
    print("STAGE 2 EDA COMPLETE")
    print("=" * 80)
    print(f"Tables saved to: {out_tables}")
    print(f"Figures saved to: {out_figures}")

    print("\nKey summary:")
    print(f"Total image files checked: {len(full_df)}")
    print(f"Readable image-mask pairs: {len(ok_df)}")
    print(f"Missing masks: {int((full_df['status'] == 'missing_mask').sum())}")
    print(f"Corrupt images: {int((full_df['status'] == 'corrupt_image').sum())}")
    print(f"Corrupt masks: {int((full_df['status'] == 'corrupt_mask').sum())}")

    if not ok_df.empty:
        print(
            "Images with tumour pixels using threshold >= 128:",
            int(ok_df["has_tumour_threshold128"].sum()),
        )
        print(
            "Empty masks using threshold >= 128:",
            int((ok_df["has_tumour_threshold128"] == False).sum()),
        )
        print(f"Mean tumour ratio: {ok_df['tumour_ratio_threshold128'].mean():.6f}")
        print(f"Median tumour ratio: {ok_df['tumour_ratio_threshold128'].median():.6f}")
        print(f"Min tumour ratio: {ok_df['tumour_ratio_threshold128'].min():.6f}")
        print(f"Max tumour ratio: {ok_df['tumour_ratio_threshold128'].max():.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="BRISC Stage 2 EDA: graphs and threshold-corrected mask statistics"
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=(
            "Path to BRISC dataset root. This should be the folder containing "
            "segmentation_task or segmentation_mask."
        ),
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output root folder for tables and figures.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    run_stage2_analysis(root=root, out_root=out_root)


if __name__ == "__main__":
    main()