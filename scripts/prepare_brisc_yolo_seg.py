from pathlib import Path
import argparse
import random
import re
import shutil
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


CLASS_MAP = {
    "gl": 0,  # glioma
    "me": 1,  # meningioma
    "pi": 2,  # pituitary
}

CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary",
}

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
    brisc2025_test_00010_me_co_t1.jpg
    """

    pattern = r"brisc2025_(train|test)_(\d+)_(gl|me|pi|nt)_(ax|co|sa)_(t1)"
    match = re.search(pattern, filename.lower())

    if match is None:
        return None

    split, index, tumour_code, plane_code, sequence = match.groups()

    return {
        "original_split": split,
        "index": int(index),
        "tumour_code": tumour_code,
        "tumour_label": TUMOUR_CODE_TO_LABEL[tumour_code],
        "plane_code": plane_code,
        "plane_label": PLANE_CODE_TO_LABEL[plane_code],
        "sequence": sequence.upper(),
    }


def find_segmentation_root(root: Path):
    """
    Supports:
    root/segmentation_task/
    root/segmentation_mask/
    root/segmentation/
    or root directly if it already contains train/images and train/masks.
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

        has_images_masks_split_structure = any(
            (candidate / "images" / split).exists()
            and (candidate / "masks" / split).exists()
            for split in ["train", "test", "val", "valid", "validation"]
        )

        has_flat_structure = (
            (candidate / "images").exists()
            and (candidate / "masks").exists()
        )

        if has_split_structure or has_images_masks_split_structure or has_flat_structure:
            return candidate

    raise FileNotFoundError(
        "Could not find a valid BRISC segmentation folder. Expected one of:\n"
        "1) root/segmentation_task/train/images and root/segmentation_task/train/masks\n"
        "2) root/segmentation_task/images/train and root/segmentation_task/masks/train\n"
        "3) root/segmentation_task/images and root/segmentation_task/masks"
    )


def find_split_dirs(root: Path):
    segmentation_root = find_segmentation_root(root)

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

    # Structure B: segmentation_task/images/train, segmentation_task/masks/train
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

    # Structure C: segmentation_task/images, segmentation_task/masks
    image_dir = segmentation_root / "images"
    mask_dir = segmentation_root / "masks"

    if image_dir.exists() and mask_dir.exists():
        split_dirs["all"] = {
            "image_dir": image_dir,
            "mask_dir": mask_dir,
        }
        return segmentation_root, split_dirs

    raise FileNotFoundError(f"No valid image/mask folders found inside: {segmentation_root}")


def collect_images(image_dir: Path):
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])


def read_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def contour_to_yolo_polygon(contour, width: int, height: int):
    """
    Converts an OpenCV contour into YOLO segmentation polygon format.
    YOLO needs:
    class_id x1 y1 x2 y2 x3 y3 ...
    Coordinates must be normalised between 0 and 1.
    """

    contour = contour.reshape(-1, 2)

    if len(contour) < 3:
        return None

    polygon = []

    for x, y in contour:
        x_norm = float(x) / float(width)
        y_norm = float(y) / float(height)

        x_norm = min(max(x_norm, 0.0), 1.0)
        y_norm = min(max(y_norm, 0.0), 1.0)

        polygon.extend([x_norm, y_norm])

    if len(polygon) < 6:
        return None

    return polygon


def mask_to_yolo_polygons(
    mask_path: Path,
    class_id: int,
    threshold: int = 128,
    min_area: float = 20.0,
    epsilon_ratio: float = 0.002,
):
    """
    Converts a binary/anti-aliased PNG mask into YOLO polygon labels.
    Mask values >= threshold are treated as tumour.
    """

    mask = read_gray(mask_path)

    if mask is None:
        return None, "corrupt_mask"

    height, width = mask.shape

    binary_mask = (mask >= threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    yolo_lines = []
    kept_contours = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        polygon = contour_to_yolo_polygon(approx, width=width, height=height)

        if polygon is None:
            continue

        values = [str(class_id)] + [f"{point:.6f}" for point in polygon]
        yolo_lines.append(" ".join(values))
        kept_contours += 1

    if kept_contours == 0:
        return [], "empty_or_too_small_mask"

    return yolo_lines, "ok"


def make_records_from_split(split_name: str, image_dir: Path, mask_dir: Path):
    image_paths = collect_images(image_dir)

    mask_suffixes = [".png", ".jpg", ".jpeg"]
    all_masks = []
    for suffix in mask_suffixes:
        all_masks.extend(mask_dir.rglob(f"*{suffix}"))

    mask_by_stem = {p.stem: p for p in all_masks}

    records = []

    for image_path in image_paths:
        parsed = parse_brisc_filename(image_path.name)

        if parsed is None:
            records.append({
                "image_path": str(image_path),
                "mask_path": None,
                "source_split": split_name,
                "status": "filename_parse_failed",
            })
            continue

        tumour_code = parsed["tumour_code"]

        if tumour_code not in CLASS_MAP:
            records.append({
                "image_path": str(image_path),
                "mask_path": None,
                "source_split": split_name,
                **parsed,
                "status": "unsupported_class_for_segmentation",
            })
            continue

        mask_path = mask_by_stem.get(image_path.stem)

        if mask_path is None:
            records.append({
                "image_path": str(image_path),
                "mask_path": None,
                "source_split": split_name,
                **parsed,
                "class_id": CLASS_MAP[tumour_code],
                "status": "missing_mask",
            })
            continue

        records.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "source_split": split_name,
            **parsed,
            "class_id": CLASS_MAP[tumour_code],
            "status": "ok",
        })

    return records


def create_train_val_split(train_df: pd.DataFrame, val_ratio: float, seed: int):
    """
    Stratifies by tumour class and MRI plane.
    This keeps glioma/meningioma/pituitary and axial/coronal/sagittal balanced.
    """

    train_df = train_df.copy()
    train_df["stratify_key"] = train_df["tumour_label"] + "_" + train_df["plane_label"]

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for _, group in train_df.groupby("stratify_key"):
        indices = group.index.to_list()
        rng.shuffle(indices)

        n_val = int(round(len(indices) * val_ratio))

        if len(indices) > 1:
            n_val = max(1, n_val)

        n_val = min(n_val, len(indices) - 1) if len(indices) > 1 else 0

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    new_train_df = train_df.loc[train_indices].copy()
    val_df = train_df.loc[val_indices].copy()

    new_train_df["split"] = "train"
    val_df["split"] = "val"

    return new_train_df, val_df


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_label_file(label_path: Path, yolo_lines):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        if yolo_lines:
            f.write("\n".join(yolo_lines))


def prepare_yolo_dataset(
    full_df: pd.DataFrame,
    output_root: Path,
    threshold: int,
    min_area: float,
    epsilon_ratio: float,
):
    rows = []
    skipped_rows = []

    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Converting masks to YOLO labels"):
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        split = row["split"]
        class_id = int(row["class_id"])

        output_image_path = output_root / "images" / split / image_path.name
        output_label_path = output_root / "labels" / split / f"{image_path.stem}.txt"

        yolo_lines, label_status = mask_to_yolo_polygons(
            mask_path=mask_path,
            class_id=class_id,
            threshold=threshold,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
        )

        if label_status != "ok":
            skipped = row.to_dict()
            skipped["label_status"] = label_status
            skipped_rows.append(skipped)
            continue

        copy_image(image_path, output_image_path)
        write_label_file(output_label_path, yolo_lines)

        converted = row.to_dict()
        converted["output_image_path"] = str(output_image_path)
        converted["output_label_path"] = str(output_label_path)
        converted["num_polygons"] = len(yolo_lines)
        converted["label_status"] = label_status
        rows.append(converted)

    converted_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped_rows)

    return converted_df, skipped_df


def write_yaml(output_root: Path):
    yaml_path = output_root / "brisc_yolo_seg.yaml"

    yaml_text = f"""# BRISC 2025 YOLO segmentation dataset
# Classes:
# 0 = glioma
# 1 = meningioma
# 2 = pituitary

path: {output_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: glioma
  1: meningioma
  2: pituitary
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    return yaml_path


def save_reports(converted_df: pd.DataFrame, skipped_df: pd.DataFrame, output_root: Path, splits_out: Path):
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    converted_df.to_csv(reports_dir / "brisc_yolo_converted_samples.csv", index=False)
    skipped_df.to_csv(reports_dir / "brisc_yolo_skipped_samples.csv", index=False)

    for split in ["train", "val", "test"]:
        split_df = converted_df[converted_df["split"] == split].copy()
        split_df.to_csv(splits_out / f"brisc_yolo_{split}.csv", index=False)

    summary = {
        "total_converted_samples": int(len(converted_df)),
        "total_skipped_samples": int(len(skipped_df)),
        "train_samples": int((converted_df["split"] == "train").sum()),
        "val_samples": int((converted_df["split"] == "val").sum()),
        "test_samples": int((converted_df["split"] == "test").sum()),
        "glioma_samples": int((converted_df["tumour_label"] == "glioma").sum()),
        "meningioma_samples": int((converted_df["tumour_label"] == "meningioma").sum()),
        "pituitary_samples": int((converted_df["tumour_label"] == "pituitary").sum()),
    }

    pd.DataFrame([summary]).to_csv(reports_dir / "brisc_yolo_conversion_summary.csv", index=False)

    split_distribution = (
        converted_df.groupby(["split"])
        .size()
        .reset_index(name="count")
        .sort_values("split")
    )
    split_distribution.to_csv(reports_dir / "brisc_yolo_split_distribution.csv", index=False)

    class_distribution = (
        converted_df.groupby(["split", "tumour_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "tumour_label"])
    )
    class_distribution.to_csv(reports_dir / "brisc_yolo_class_distribution.csv", index=False)

    plane_distribution = (
        converted_df.groupby(["split", "plane_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "plane_label"])
    )
    plane_distribution.to_csv(reports_dir / "brisc_yolo_plane_distribution.csv", index=False)

    return reports_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BRISC 2025 segmentation dataset for YOLO segmentation training."
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to BRISC dataset root folder. It should contain segmentation_task or segmentation_mask.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for YOLO-ready dataset, e.g. data/processed/brisc_yolo_seg",
    )

    parser.add_argument(
        "--splits-out",
        type=str,
        default="data/splits",
        help="Folder where split CSV files will be saved.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio taken from the original training split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Mask threshold. Pixels >= threshold are treated as tumour.",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Minimum contour area in pixels. Smaller contours are ignored.",
    )

    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.002,
        help="Polygon simplification ratio. Lower value keeps more contour detail.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.out)
    splits_out = Path(args.splits_out)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    segmentation_root, split_dirs = find_split_dirs(root)

    print("=" * 80)
    print("BRISC TO YOLO SEGMENTATION PREPARATION")
    print("=" * 80)
    print(f"Dataset root: {root}")
    print(f"Detected segmentation root: {segmentation_root}")
    print(f"Output YOLO dataset root: {output_root}")
    print(f"Split CSV output folder: {splits_out}")
    print(f"Mask threshold: {args.threshold}")
    print(f"Validation ratio: {args.val_ratio}")
    print("=" * 80)

    all_records = []

    for source_split, paths in split_dirs.items():
        print(f"\n[INFO] Reading source split: {source_split}")
        print(f"Images: {paths['image_dir']}")
        print(f"Masks:  {paths['mask_dir']}")

        records = make_records_from_split(
            split_name=source_split,
            image_dir=paths["image_dir"],
            mask_dir=paths["mask_dir"],
        )

        all_records.extend(records)

    raw_df = pd.DataFrame(all_records)

    if raw_df.empty:
        raise RuntimeError("No records found. Check your BRISC root path.")

    raw_report_dir = output_root / "reports"
    raw_report_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_report_dir / "brisc_yolo_raw_records.csv", index=False)

    ok_df = raw_df[raw_df["status"] == "ok"].copy()

    if ok_df.empty:
        raise RuntimeError("No valid image-mask records found.")

    if "train" not in ok_df["source_split"].unique():
        raise RuntimeError("No original train split found. Cannot create train/val split.")

    original_train_df = ok_df[ok_df["source_split"] == "train"].copy()
    original_test_df = ok_df[ok_df["source_split"] == "test"].copy()

    train_df, val_df = create_train_val_split(
        train_df=original_train_df,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    original_test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, original_test_df], ignore_index=True)

    print("\n[INFO] Final split distribution before conversion:")
    print(final_df["split"].value_counts())

    print("\n[INFO] Final class distribution before conversion:")
    print(final_df.groupby(["split", "tumour_label"]).size())

    converted_df, skipped_df = prepare_yolo_dataset(
        full_df=final_df,
        output_root=output_root,
        threshold=args.threshold,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
    )

    yaml_path = write_yaml(output_root)

    reports_dir = save_reports(
        converted_df=converted_df,
        skipped_df=skipped_df,
        output_root=output_root,
        splits_out=splits_out,
    )

    print("\n" + "=" * 80)
    print("YOLO DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"YOLO dataset saved to: {output_root}")
    print(f"YOLO YAML file: {yaml_path}")
    print(f"Reports saved to: {reports_dir}")
    print(f"Split CSVs saved to: {splits_out}")

    print("\nConverted sample count:")
    print(converted_df["split"].value_counts())

    print("\nConverted class distribution:")
    print(converted_df.groupby(["split", "tumour_label"]).size())

    if not skipped_df.empty:
        print("\n[WARNING] Some samples were skipped:")
        print(skipped_df["label_status"].value_counts())
        print(f"Check: {reports_dir / 'brisc_yolo_skipped_samples.csv'}")


if __name__ == "__main__":
    main()