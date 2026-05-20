from pathlib import Path
import argparse
import re
import os
import gc

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion

from ultralytics import YOLO


EPS = 1e-7


# -----------------------------
# Basic utilities
# -----------------------------

def read_mask_binary(mask_path: Path, threshold: int = 128):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    return mask >= threshold


def get_eval_image_path(row):
    """
    Prefer YOLO processed image path if available.
    Fall back to original image path.
    """

    for col in ["output_image_path", "image_path"]:
        if col in row and pd.notna(row[col]):
            path = Path(str(row[col]))
            if path.exists():
                return path

    raise FileNotFoundError("No valid image path found in validation CSV row.")


def extract_epoch_number(weight_path: Path):
    match = re.search(r"epoch(\d+)", weight_path.stem.lower())

    if match:
        return int(match.group(1))

    return None


def find_epoch_weights(run_dir: Path):
    weights_dir = run_dir / "weights"

    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights folder not found: {weights_dir}")

    epoch_weights = list(weights_dir.glob("epoch*.pt"))

    epoch_weights = [
        p for p in epoch_weights
        if extract_epoch_number(p) is not None
    ]

    epoch_weights = sorted(epoch_weights, key=lambda p: extract_epoch_number(p))

    return epoch_weights


# -----------------------------
# Prediction mask conversion
# -----------------------------

def result_to_binary_mask(result, target_shape):
    """
    Converts YOLO segmentation prediction into a binary mask.
    All predicted tumour masks are combined into one binary prediction mask.
    """

    height, width = target_shape
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
    Returns top-confidence predicted class.
    If no detection exists, returns -1.
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


# -----------------------------
# Medical segmentation metrics
# -----------------------------

def compute_overlap_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
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

    dice_loss = 1.0 - dice

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

    hd = float(np.max(distances))
    hd95 = float(np.percentile(distances, 95))
    asd = float(np.mean(distances))

    return {
        "hd": hd,
        "hd95": hd95,
        "asd": asd,
    }


def compute_all_metrics(pred, gt):
    overlap = compute_overlap_metrics(pred, gt)
    boundary = compute_boundary_metrics(pred, gt)

    metrics = {}
    metrics.update(overlap)
    metrics.update(boundary)

    return metrics


# -----------------------------
# Epoch evaluation
# -----------------------------

def evaluate_weight_on_val(
    weight_path: Path,
    val_df: pd.DataFrame,
    epoch: int,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    eval_batch: int,
    mask_threshold: int,
    eval_img_limit: int = 0,
    save_per_sample: bool = False,
    per_sample_dir: Path | None = None,
):
    model = YOLO(str(weight_path))

    if eval_img_limit and eval_img_limit > 0:
        eval_df = val_df.head(eval_img_limit).copy()
    else:
        eval_df = val_df.copy()

    image_paths = [str(get_eval_image_path(row)) for _, row in eval_df.iterrows()]

    results_generator = model.predict(
        source=image_paths,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        batch=eval_batch,
        stream=True,
        verbose=False,
    )

    sample_rows = []

    for (_, row), result in tqdm(
        zip(eval_df.iterrows(), results_generator),
        total=len(eval_df),
        desc=f"Evaluating epoch {epoch}",
        leave=False,
    ):
        mask_path = Path(str(row["mask_path"]))
        gt_mask = read_mask_binary(mask_path, threshold=mask_threshold)

        pred_mask = result_to_binary_mask(result, target_shape=gt_mask.shape)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        metrics = compute_all_metrics(pred_mask, gt_mask)

        gt_class = int(row["class_id"])
        pred_class = get_top_predicted_class(result)
        class_correct = int(pred_class == gt_class)

        sample_result = {
            "epoch": epoch,
            "image_filename": Path(str(row["image_path"])).name,
            "mask_filename": Path(str(row["mask_path"])).name,
            "tumour_label": row.get("tumour_label", "unknown"),
            "plane_label": row.get("plane_label", "unknown"),
            "gt_class_id": gt_class,
            "top_pred_class_id": pred_class,
            "class_correct": class_correct,
            **metrics,
        }

        sample_rows.append(sample_result)

    sample_df = pd.DataFrame(sample_rows)

    if save_per_sample and per_sample_dir is not None:
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(per_sample_dir / f"val_sample_metrics_epoch_{epoch:03d}.csv", index=False)

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
    ]

    summary = {
        "epoch": epoch,
        "weight_path": str(weight_path),
        "num_val_images": int(len(sample_df)),
    }

    for col in metric_cols:
        summary[f"val_{col}"] = float(sample_df[col].mean())

    summary["val_class_accuracy"] = float(sample_df["class_correct"].mean())

    del model
    gc.collect()

    return summary


# -----------------------------
# Training
# -----------------------------

def train_yolo(args):
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=1,
        patience=0,
        workers=args.workers,
        optimizer=args.optimizer,
        plots=True,
        verbose=True,
    )


def merge_with_ultralytics_results(run_dir: Path, custom_metrics_path: Path):
    ultralytics_results_path = run_dir / "results.csv"

    if not ultralytics_results_path.exists():
        print(f"[WARNING] Ultralytics results.csv not found: {ultralytics_results_path}")
        return None

    train_df = pd.read_csv(ultralytics_results_path)
    train_df.columns = [c.strip() for c in train_df.columns]

    custom_df = pd.read_csv(custom_metrics_path)

    if "epoch" not in train_df.columns:
        print("[WARNING] No epoch column found in Ultralytics results.csv")
        return None

    train_df["epoch_for_merge"] = train_df["epoch"].astype(int)

    # Ultralytics sometimes stores epochs as 0-based.
    if train_df["epoch_for_merge"].min() == 0:
        train_df["epoch_for_merge"] = train_df["epoch_for_merge"] + 1

    merged_df = train_df.merge(
        custom_df,
        left_on="epoch_for_merge",
        right_on="epoch",
        how="left",
        suffixes=("_ultralytics", "_custom"),
    )

    merged_path = run_dir / "research_metrics_merged.csv"
    merged_df.to_csv(merged_path, index=False)

    return merged_path


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv26 segmentation on BRISC and compute research metrics per epoch."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO segmentation model weights, e.g. yolo26n-seg.pt",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO dataset yaml file.",
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        required=True,
        help="Path to validation split CSV created during YOLO dataset preparation.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="experiments/brisc_yolo26",
        help="Experiment project folder.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo26n_multiclass_seg",
        help="Experiment name.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="auto")

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for validation prediction metrics.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for validation prediction metrics.",
    )

    parser.add_argument(
        "--eval-batch",
        type=int,
        default=4,
        help="Batch size during prediction-based metric evaluation.",
    )

    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=128,
        help="Ground-truth mask threshold. Pixels >= threshold are tumour.",
    )

    parser.add_argument(
        "--eval-img-limit",
        type=int,
        default=0,
        help="Use 0 for full validation set. Use small number like 50 for debugging.",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only evaluate saved epoch weights.",
    )

    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Save per-image metrics for every epoch. This creates many CSV rows.",
    )

    parser.add_argument(
        "--delete-epoch-weights-after-eval",
        action="store_true",
        help="Delete epoch*.pt files after custom metrics are computed. Keeps last.pt and best.pt.",
    )

    args = parser.parse_args()

    run_dir = Path(args.project) / args.name
    val_csv = Path(args.val_csv)

    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    val_df = pd.read_csv(val_csv)

    required_cols = ["image_path", "mask_path", "class_id"]

    for col in required_cols:
        if col not in val_df.columns:
            raise ValueError(f"Validation CSV missing required column: {col}")

    print("=" * 80)
    print("YOLOv26 BRISC RESEARCH TRAINING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data YAML: {args.data}")
    print(f"Validation CSV: {val_csv}")
    print(f"Run directory: {run_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print("=" * 80)

    if not args.skip_train:
        train_yolo(args)
    else:
        print("[INFO] Skipping training. Evaluating existing epoch weights only.")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found after training: {run_dir}")

    epoch_weights = find_epoch_weights(run_dir)

    if len(epoch_weights) == 0:
        raise RuntimeError(
            "No epoch*.pt weights found. Make sure training used save_period=1."
        )

    print("\n" + "=" * 80)
    print("CUSTOM VALIDATION METRICS PER EPOCH")
    print("=" * 80)
    print(f"Epoch weights found: {len(epoch_weights)}")

    custom_rows = []
    per_sample_dir = run_dir / "per_sample_val_metrics"

    for weight_path in epoch_weights:
        epoch = extract_epoch_number(weight_path)

        summary = evaluate_weight_on_val(
            weight_path=weight_path,
            val_df=val_df,
            epoch=epoch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            eval_batch=args.eval_batch,
            mask_threshold=args.mask_threshold,
            eval_img_limit=args.eval_img_limit,
            save_per_sample=args.save_per_sample,
            per_sample_dir=per_sample_dir,
        )

        custom_rows.append(summary)

        print(
            f"Epoch {epoch:03d} | "
            f"Dice: {summary['val_dice']:.4f} | "
            f"Dice Loss: {summary['val_dice_loss']:.4f} | "
            f"IoU: {summary['val_iou']:.4f} | "
            f"HD95: {summary['val_hd95']:.4f} | "
            f"ASD: {summary['val_asd']:.4f} | "
            f"Class Acc: {summary['val_class_accuracy']:.4f}"
        )

        if args.delete_epoch_weights_after_eval:
            try:
                os.remove(weight_path)
            except OSError:
                pass

    custom_df = pd.DataFrame(custom_rows).sort_values("epoch")

    custom_metrics_path = run_dir / "custom_val_metrics_by_epoch.csv"
    custom_df.to_csv(custom_metrics_path, index=False)

    merged_path = merge_with_ultralytics_results(
        run_dir=run_dir,
        custom_metrics_path=custom_metrics_path,
    )

    print("\n" + "=" * 80)
    print("RESEARCH METRICS COMPLETE")
    print("=" * 80)
    print(f"Custom metrics saved to: {custom_metrics_path}")

    if merged_path is not None:
        print(f"Merged training + custom metrics saved to: {merged_path}")

    best_row = custom_df.loc[custom_df["val_dice"].idxmax()]

    print("\nBest epoch by validation Dice:")
    print(f"Epoch: {int(best_row['epoch'])}")
    print(f"Dice: {best_row['val_dice']:.4f}")
    print(f"Dice Loss: {best_row['val_dice_loss']:.4f}")
    print(f"IoU: {best_row['val_iou']:.4f}")
    print(f"HD95: {best_row['val_hd95']:.4f}")
    print(f"ASD: {best_row['val_asd']:.4f}")
    print(f"Class Accuracy: {best_row['val_class_accuracy']:.4f}")


if __name__ == "__main__":
    main()