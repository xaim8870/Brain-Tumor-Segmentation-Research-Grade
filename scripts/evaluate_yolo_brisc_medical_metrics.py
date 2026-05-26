from pathlib import Path
import argparse
import re
import gc

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
from ultralytics import YOLO


EPS = 1e-7


# -------------------------------------------------------
# File and weight utilities
# -------------------------------------------------------

def extract_epoch_number(weight_path: Path):
    """
    Extracts epoch number from names like:
    epoch1.pt, epoch25.pt, epoch100.pt
    """
    match = re.search(r"epoch(\d+)", weight_path.stem.lower())
    if match:
        return int(match.group(1))
    return None


def collect_weights(weights_dir: Path, mode: str):
    """
    mode options:
    - all: all epoch*.pt if available, otherwise best.pt and last.pt
    - best: only best.pt
    - last: only last.pt
    - best-last: best.pt and last.pt
    - epochs: only epoch*.pt
    """

    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"

    epoch_weights = sorted(
        [p for p in weights_dir.glob("epoch*.pt") if extract_epoch_number(p) is not None],
        key=lambda p: extract_epoch_number(p)
    )

    selected = []

    if mode == "best":
        if best.exists():
            selected = [best]

    elif mode == "last":
        if last.exists():
            selected = [last]

    elif mode == "best-last":
        if best.exists():
            selected.append(best)
        if last.exists():
            selected.append(last)

    elif mode == "epochs":
        selected = epoch_weights

    elif mode == "all":
        if epoch_weights:
            selected = epoch_weights
        else:
            if best.exists():
                selected.append(best)
            if last.exists():
                selected.append(last)

    if not selected:
        raise RuntimeError(
            f"No weights found for mode='{mode}' in {weights_dir}. "
            f"Available files: {[p.name for p in weights_dir.glob('*.pt')]}"
        )

    return selected


def get_weight_name(weight_path: Path):
    epoch = extract_epoch_number(weight_path)

    if epoch is not None:
        return f"epoch_{epoch:03d}", epoch

    if weight_path.stem.lower() == "best":
        return "best", -1

    if weight_path.stem.lower() == "last":
        return "last", -2

    return weight_path.stem, -999


# -------------------------------------------------------
# Data utilities
# -------------------------------------------------------
def read_mask_binary(mask_path: Path, threshold: int = 128):
    """
    Reads a ground-truth mask and converts it into a 2D binary mask.

    Handles:
    - grayscale masks: H x W
    - single-channel masks: H x W x 1
    - RGB/BGR masks: H x W x 3
    - RGBA/BGRA masks: H x W x 4

    Output:
    - shape: H x W
    - dtype: bool
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise FileNotFoundError(f"Could not read ground-truth mask: {mask_path}")

    # Case 1: already grayscale, shape = H x W
    if mask.ndim == 2:
        pass

    # Case 2: has channel dimension, shape = H x W x C
    elif mask.ndim == 3:
        channels = mask.shape[2]

        # H x W x 1
        if channels == 1:
            mask = mask[:, :, 0]

        # H x W x 3
        elif channels == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # H x W x 4
        elif channels == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)

        else:
            raise ValueError(
                f"Unsupported mask channel count: {channels}, shape={mask.shape}, path={mask_path}"
            )

    else:
        raise ValueError(
            f"Unsupported mask shape: {mask.shape}, path={mask_path}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Mask should be 2D after processing, but got shape {mask.shape}: {mask_path}"
        )

    return mask >= threshold


def get_image_path_from_row(row):
    """
    The split CSV may contain:
    - output_image_path
    - image_path

    We prefer output_image_path because that is the YOLO-ready image.
    """

    candidate_cols = ["output_image_path", "image_path"]

    for col in candidate_cols:
        if col in row and pd.notna(row[col]):
            path = Path(str(row[col]))
            if path.exists():
                return path

    raise FileNotFoundError(
        f"No valid image path found for row. Checked columns: {candidate_cols}"
    )


def get_mask_path_from_row(row):
    if "mask_path" not in row or pd.isna(row["mask_path"]):
        raise FileNotFoundError("mask_path column missing or empty in CSV row.")

    mask_path = Path(str(row["mask_path"]))

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {mask_path}")

    return mask_path


# -------------------------------------------------------
# YOLO prediction conversion
# -------------------------------------------------------

def result_to_binary_mask(result, target_shape):
    """
    Converts YOLO predicted segmentation polygons into a 2D binary mask.

    For medical metrics, YOLO polygon predictions must be converted back into
    pixel masks so they can be compared with the ground-truth binary mask.
    """

    if len(target_shape) == 2:
        height, width = target_shape
    elif len(target_shape) == 3:
        height, width = target_shape[:2]
    else:
        raise ValueError(f"Invalid target shape: {target_shape}")

    pred_mask = np.zeros((height, width), dtype=np.uint8)

    if result.masks is None:
        return pred_mask.astype(bool)

    if result.masks.xy is None:
        return pred_mask.astype(bool)

    for polygon in result.masks.xy:
        if polygon is None or len(polygon) < 3:
            continue

        points = np.asarray(polygon, dtype=np.float32)

        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

        points = points.astype(np.int32)

        if len(points) >= 3:
            cv2.fillPoly(pred_mask, [points], 1)

    return pred_mask.astype(bool)

def get_top_predicted_class(result):
    """
    Returns the predicted class with highest confidence.
    If no object is predicted, returns -1.
    """

    if result.boxes is None:
        return -1

    if result.boxes.cls is None or result.boxes.conf is None:
        return -1

    if len(result.boxes.cls) == 0:
        return -1

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()

    top_index = int(np.argmax(confs))
    return int(classes[top_index])


def get_num_predictions(result):
    if result.boxes is None or result.boxes.cls is None:
        return 0
    return int(len(result.boxes.cls))


# -------------------------------------------------------
# Medical segmentation metrics
# -------------------------------------------------------

def compute_overlap_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    dice_loss = 1.0 - dice

    iou = tp / (tp + fp + fn + EPS)

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)

    pred_volume = pred.sum()
    gt_volume = gt.sum()

    volume_similarity = 1.0 - (
        abs(float(pred_volume) - float(gt_volume)) /
        (float(pred_volume) + float(gt_volume) + EPS)
    )

    return {
        "dice": float(dice),
        "dice_loss": float(dice_loss),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "volume_similarity": float(volume_similarity),
        "pred_pixels": int(pred_volume),
        "gt_pixels": int(gt_volume),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def get_surface(mask):
    mask = mask.astype(bool)

    if not mask.any():
        return mask

    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    surface = np.logical_xor(mask, eroded)

    if not surface.any():
        return mask

    return surface


def compute_surface_distances(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    height, width = gt.shape
    max_distance = float(np.sqrt(height ** 2 + width ** 2))

    if not pred.any() and not gt.any():
        return np.array([0.0], dtype=np.float32)

    if pred.any() != gt.any():
        return np.array([max_distance], dtype=np.float32)

    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    dt_gt = distance_transform_edt(np.logical_not(gt_surface))
    dt_pred = distance_transform_edt(np.logical_not(pred_surface))

    pred_to_gt = dt_gt[pred_surface]
    gt_to_pred = dt_pred[gt_surface]

    distances = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float32)

    if distances.size == 0:
        return np.array([0.0], dtype=np.float32)

    return distances


def compute_boundary_metrics(pred, gt):
    distances = compute_surface_distances(pred, gt)

    return {
        "hd": float(np.max(distances)),
        "hd95": float(np.percentile(distances, 95)),
        "asd": float(np.mean(distances)),
    }


def compute_all_metrics(pred, gt):
    metrics = {}
    metrics.update(compute_overlap_metrics(pred, gt))
    metrics.update(compute_boundary_metrics(pred, gt))
    return metrics


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate_one_weight(
    weight_path: Path,
    data_df: pd.DataFrame,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch: int,
    mask_threshold: int,
    limit: int,
    save_predictions: bool,
    predictions_dir: Path,
):
    weight_name, epoch = get_weight_name(weight_path)

    model = YOLO(str(weight_path))

    if limit and limit > 0:
        eval_df = data_df.head(limit).copy()
    else:
        eval_df = data_df.copy()

    image_paths = [str(get_image_path_from_row(row)) for _, row in eval_df.iterrows()]

    results_generator = model.predict(
        source=image_paths,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        batch=batch,
        stream=True,
        verbose=False,
    )

    sample_rows = []

    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for (_, row), result in tqdm(
        zip(eval_df.iterrows(), results_generator),
        total=len(eval_df),
        desc=f"Evaluating {weight_path.name}",
    ):
        image_path = get_image_path_from_row(row)
        mask_path = get_mask_path_from_row(row)

        gt_mask = read_mask_binary(mask_path, threshold=mask_threshold)
        pred_mask = result_to_binary_mask(result, target_shape=gt_mask.shape)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        metrics = compute_all_metrics(pred_mask, gt_mask)

        gt_class = int(row["class_id"]) if "class_id" in row and pd.notna(row["class_id"]) else -1
        pred_class = get_top_predicted_class(result)

        class_correct = int(pred_class == gt_class)
        num_predictions = get_num_predictions(result)

        sample_result = {
            "weight_name": weight_name,
            "epoch": epoch,
            "weight_path": str(weight_path),
            "image_filename": image_path.name,
            "mask_filename": mask_path.name,
            "tumour_label": row.get("tumour_label", "unknown"),
            "plane_label": row.get("plane_label", "unknown"),
            "gt_class_id": gt_class,
            "top_pred_class_id": pred_class,
            "class_correct": class_correct,
            "num_predictions": num_predictions,
            **metrics,
        }

        sample_rows.append(sample_result)

        if save_predictions:
            pred_save_path = predictions_dir / weight_name / f"{image_path.stem}_pred_mask.png"
            pred_save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pred_save_path), pred_mask.astype(np.uint8) * 255)

    sample_df = pd.DataFrame(sample_rows)

    metric_cols = [
        "dice",
        "dice_loss",
        "iou",
        "precision",
        "recall",
        "specificity",
        "volume_similarity",
        "hd",
        "hd95",
        "asd",
        "pred_pixels",
        "gt_pixels",
        "tp",
        "fp",
        "fn",
        "tn",
        "class_correct",
        "num_predictions",
    ]

    summary = {
        "weight_name": weight_name,
        "epoch": epoch,
        "weight_path": str(weight_path),
        "num_images": int(len(sample_df)),
    }

    for col in metric_cols:
        summary[col] = float(sample_df[col].mean())

    # Add class-wise Dice, IoU and HD95
    for tumour_label, group in sample_df.groupby("tumour_label"):
        clean_label = str(tumour_label).replace(" ", "_").lower()
        summary[f"{clean_label}_dice"] = float(group["dice"].mean())
        summary[f"{clean_label}_iou"] = float(group["iou"].mean())
        summary[f"{clean_label}_hd95"] = float(group["hd95"].mean())

    del model
    gc.collect()

    return summary, sample_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO BRISC segmentation weights using medical segmentation metrics."
    )

    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Path to YOLO weights folder containing best.pt, last.pt, or epoch*.pt.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to split CSV, e.g. data/splits/brisc_yolo_val.csv or brisc_yolo_test.csv.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/yolo_brisc_medical_eval",
        help="Output folder for medical metric reports.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="best-last",
        choices=["all", "best", "last", "best-last", "epochs"],
        help="Which weights to evaluate.",
    )

    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--mask-threshold", type=int, default=128)

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use 0 for full evaluation. Use small number like 50 for debugging.",
    )

    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Save per-image metrics CSV.",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted binary masks as PNG files.",
    )

    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    data_df = pd.read_csv(csv_path)

    required_cols = ["mask_path", "class_id"]

    for col in required_cols:
        if col not in data_df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    if "output_image_path" not in data_df.columns and "image_path" not in data_df.columns:
        raise ValueError("CSV must contain either output_image_path or image_path column.")

    weights = collect_weights(weights_dir, mode=args.mode)

    print("=" * 80)
    print("YOLO BRISC MEDICAL SEGMENTATION EVALUATION")
    print("=" * 80)
    print(f"Weights directory: {weights_dir}")
    print(f"CSV file: {csv_path}")
    print(f"Output directory: {out_dir}")
    print(f"Weights selected: {[w.name for w in weights]}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf}")
    print(f"NMS IoU threshold: {args.iou}")
    print(f"Device: {args.device}")
    print("=" * 80)

    summary_rows = []
    all_sample_dfs = []

    predictions_dir = out_dir / "predicted_masks"

    for weight_path in weights:
        summary, sample_df = evaluate_one_weight(
            weight_path=weight_path,
            data_df=data_df,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=args.batch,
            mask_threshold=args.mask_threshold,
            limit=args.limit,
            save_predictions=args.save_predictions,
            predictions_dir=predictions_dir,
        )

        summary_rows.append(summary)

        if args.save_per_sample:
            all_sample_dfs.append(sample_df)

        print(
            f"{summary['weight_name']} | "
            f"Dice: {summary['dice']:.4f} | "
            f"Dice Loss: {summary['dice_loss']:.4f} | "
            f"IoU: {summary['iou']:.4f} | "
            f"Precision: {summary['precision']:.4f} | "
            f"Recall: {summary['recall']:.4f} | "
            f"HD95: {summary['hd95']:.4f} | "
            f"ASD: {summary['asd']:.4f} | "
            f"Class Acc: {summary['class_correct']:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)

    # Sort: epochs first in order, then best/last if present
    summary_df = summary_df.sort_values(["epoch", "weight_name"])

    summary_path = out_dir / "medical_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.save_per_sample and all_sample_dfs:
        per_sample_df = pd.concat(all_sample_dfs, ignore_index=True)
        per_sample_path = out_dir / "medical_metrics_per_sample.csv"
        per_sample_df.to_csv(per_sample_path, index=False)
    else:
        per_sample_path = None

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Summary metrics saved to: {summary_path}")

    if per_sample_path is not None:
        print(f"Per-sample metrics saved to: {per_sample_path}")

    # Best by Dice
    best_idx = summary_df["dice"].idxmax()
    best_row = summary_df.loc[best_idx]

    print("\nBest weight by Dice:")
    print(f"Weight: {best_row['weight_name']}")
    print(f"Dice: {best_row['dice']:.4f}")
    print(f"Dice Loss: {best_row['dice_loss']:.4f}")
    print(f"IoU: {best_row['iou']:.4f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall: {best_row['recall']:.4f}")
    print(f"Specificity: {best_row['specificity']:.4f}")
    print(f"HD95: {best_row['hd95']:.4f}")
    print(f"ASD: {best_row['asd']:.4f}")
    print(f"Volume Similarity: {best_row['volume_similarity']:.4f}")
    print(f"Class Accuracy: {best_row['class_correct']:.4f}")


if __name__ == "__main__":
    main()