# scripts/train_yolo26_brisc_research.py

from pathlib import Path
import argparse
import re
import os
import gc
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ultralytics import YOLO


# -------------------------------------------------------
# Repo imports
# -------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.metrics.segmentation_metrics import (
    STANDARD_BINARY_METRIC_COLUMNS,
    compute_binary_segmentation_metrics,
)


# -------------------------------------------------------
# Basic utilities
# -------------------------------------------------------

def read_mask_binary(mask_path: Path, threshold: int = 128):
    """
    Reads a ground-truth mask and converts it into a 2D binary mask.

    Handles:
    - H x W grayscale masks
    - H x W x 1 masks
    - H x W x 3 BGR masks
    - H x W x 4 BGRA masks

    Output:
    - H x W boolean mask
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    if mask.ndim == 2:
        pass

    elif mask.ndim == 3:
        channels = mask.shape[2]

        if channels == 1:
            mask = mask[:, :, 0]

        elif channels == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        elif channels == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)

        else:
            raise ValueError(
                f"Unsupported mask channel count: {channels}, "
                f"shape={mask.shape}, path={mask_path}"
            )

    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}, path={mask_path}")

    if mask.ndim != 2:
        raise ValueError(
            f"Mask should be 2D after processing, got {mask.shape}: {mask_path}"
        )

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

    raise FileNotFoundError(
        "No valid image path found. Expected output_image_path or image_path."
    )


def get_mask_path(row):
    if "mask_path" not in row or pd.isna(row["mask_path"]):
        raise FileNotFoundError("mask_path column missing or empty.")

    mask_path = Path(str(row["mask_path"]))

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {mask_path}")

    return mask_path


def extract_epoch_number(weight_path: Path):
    """
    Extracts epoch number from:
    epoch1.pt
    epoch25.pt
    epoch100.pt
    """

    match = re.search(r"epoch(\d+)", weight_path.stem.lower())

    if match:
        return int(match.group(1))

    return None


def get_weight_name_and_epoch(weight_path: Path):
    epoch = extract_epoch_number(weight_path)

    if epoch is not None:
        return f"epoch_{epoch:03d}", epoch

    stem = weight_path.stem.lower()

    if stem == "best":
        return "best", -1

    if stem == "last":
        return "last", -2

    return weight_path.stem, -999


def collect_weights(run_dir: Path, mode: str = "epochs"):
    """
    mode:
    - epochs: only epoch*.pt
    - best: only best.pt
    - last: only last.pt
    - best-last: best.pt and last.pt
    - all: epoch*.pt if available, otherwise best.pt and last.pt
    """

    weights_dir = run_dir / "weights"

    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights folder not found: {weights_dir}")

    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"

    epoch_weights = sorted(
        [
            p for p in weights_dir.glob("epoch*.pt")
            if extract_epoch_number(p) is not None
        ],
        key=lambda p: extract_epoch_number(p),
    )

    selected = []

    if mode == "epochs":
        selected = epoch_weights

    elif mode == "best":
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

    elif mode == "all":
        if epoch_weights:
            selected = epoch_weights
        else:
            if best.exists():
                selected.append(best)
            if last.exists():
                selected.append(last)

    if not selected:
        available = [p.name for p in weights_dir.glob("*.pt")]
        raise RuntimeError(
            f"No weights found for mode={mode}. "
            f"Available files in {weights_dir}: {available}"
        )

    return selected


def validate_split_csv(df: pd.DataFrame, csv_path: Path):
    required_cols = ["mask_path", "class_id"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing required column: {col}")

    if "output_image_path" not in df.columns and "image_path" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain either output_image_path or image_path column."
        )


def find_actual_run_dir(project: str, name: str) -> Path:
    """
    Finds where Ultralytics actually saved the run.

    It may save in:
    - experiments/brisc_yolo26/run_name
    - REPO_ROOT/experiments/brisc_yolo26/run_name
    - REPO_ROOT/runs/segment/experiments/brisc_yolo26/run_name
    - runs/segment/experiments/brisc_yolo26/run_name
    """

    project_path = Path(project)

    candidates = [
        project_path / name,
        REPO_ROOT / project_path / name,
        REPO_ROOT / "runs" / "segment" / project_path / name,
        Path("runs") / "segment" / project_path / name,
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "weights").exists():
            return candidate

    print("\n[DEBUG] Checked these possible run directories:")
    for candidate in candidates:
        print(candidate)

    raise FileNotFoundError(
        "Could not find YOLO run directory with weights folder. "
        "Check where Ultralytics saved the training run."
    )


# -------------------------------------------------------
# YOLO prediction conversion
# -------------------------------------------------------

def result_to_binary_mask(result, target_shape):
    """
    Converts YOLO segmentation polygons into a 2D binary mask.

    For medical segmentation metrics, predicted polygons must be converted
    into pixel masks before comparing with the ground-truth mask.
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
    Returns the class with highest confidence.
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


def get_num_predictions(result):
    if result.boxes is None or result.boxes.cls is None:
        return 0

    return int(len(result.boxes.cls))


# -------------------------------------------------------
# Evaluation on train / val split
# -------------------------------------------------------

def evaluate_model_on_split(
    model: YOLO,
    weight_path: Path,
    data_df: pd.DataFrame,
    split_name: str,
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
    """
    Evaluates one YOLO checkpoint on one split.

    split_name:
    - train
    - val
    - test

    Returns:
    - summary dictionary with prefixed metrics.
    """

    if eval_img_limit and eval_img_limit > 0:
        eval_df = data_df.head(eval_img_limit).copy()
    else:
        eval_df = data_df.copy()

    image_paths = [
        str(get_eval_image_path(row))
        for _, row in eval_df.iterrows()
    ]

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
        desc=f"Evaluating {split_name} epoch {epoch}",
        leave=False,
    ):
        image_path = get_eval_image_path(row)
        mask_path = get_mask_path(row)

        gt_mask = read_mask_binary(mask_path, threshold=mask_threshold)
        pred_mask = result_to_binary_mask(result, target_shape=gt_mask.shape)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        metrics = compute_binary_segmentation_metrics(
            pred=pred_mask,
            gt=gt_mask,
            pred_threshold=None,
            gt_threshold=None,
            spacing=None,
        )

        gt_class = int(row["class_id"]) if "class_id" in row and pd.notna(row["class_id"]) else -1
        pred_class = get_top_predicted_class(result)

        class_correct = int(pred_class == gt_class)
        num_predictions = get_num_predictions(result)

        sample_result = {
            "epoch": epoch,
            "split": split_name,
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

    sample_df = pd.DataFrame(sample_rows)

    if save_per_sample and per_sample_dir is not None:
        split_dir = per_sample_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(
            split_dir / f"{split_name}_sample_metrics_epoch_{epoch:03d}.csv",
            index=False,
        )

    summary = {
        f"{split_name}_num_images": int(len(sample_df)),
    }

    for col in STANDARD_BINARY_METRIC_COLUMNS:
        if col in sample_df.columns:
            summary[f"{split_name}_{col}"] = float(sample_df[col].mean())

    summary[f"{split_name}_class_accuracy"] = float(sample_df["class_correct"].mean())
    summary[f"{split_name}_avg_num_predictions"] = float(sample_df["num_predictions"].mean())

    # Class-wise metrics for tumour-type-aware segmentation
    for tumour_label, group in sample_df.groupby("tumour_label"):
        clean_label = str(tumour_label).strip().lower().replace(" ", "_")

        summary[f"{split_name}_{clean_label}_dice"] = float(group["dice"].mean())
        summary[f"{split_name}_{clean_label}_iou"] = float(group["iou"].mean())
        summary[f"{split_name}_{clean_label}_hd95"] = float(group["hd95"].mean())
        summary[f"{split_name}_{clean_label}_asd"] = float(group["asd"].mean())

    return summary


def evaluate_one_weight_on_train_and_val(
    weight_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    eval_batch: int,
    mask_threshold: int,
    train_eval_limit: int,
    val_eval_limit: int,
    save_per_sample: bool,
    per_sample_dir: Path,
    evaluate_train: bool = True,
):
    weight_name, epoch = get_weight_name_and_epoch(weight_path)

    model = YOLO(str(weight_path))

    row = {
        "epoch": epoch,
        "weight_name": weight_name,
        "weight_path": str(weight_path),
    }

    if evaluate_train:
        train_summary = evaluate_model_on_split(
            model=model,
            weight_path=weight_path,
            data_df=train_df,
            split_name="train",
            epoch=epoch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            eval_batch=eval_batch,
            mask_threshold=mask_threshold,
            eval_img_limit=train_eval_limit,
            save_per_sample=save_per_sample,
            per_sample_dir=per_sample_dir,
        )
        row.update(train_summary)

    val_summary = evaluate_model_on_split(
        model=model,
        weight_path=weight_path,
        data_df=val_df,
        split_name="val",
        epoch=epoch,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        eval_batch=eval_batch,
        mask_threshold=mask_threshold,
        eval_img_limit=val_eval_limit,
        save_per_sample=save_per_sample,
        per_sample_dir=per_sample_dir,
    )
    row.update(val_summary)

    del model
    gc.collect()

    return row


# -------------------------------------------------------
# YOLO training
# -------------------------------------------------------

def train_yolo(args):
    model = YOLO(args.model)

    results = model.train(
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

    save_dir = None

    if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
        save_dir = Path(model.trainer.save_dir)

    elif hasattr(results, "save_dir"):
        save_dir = Path(results.save_dir)

    return save_dir


# -------------------------------------------------------
# CSV merge
# -------------------------------------------------------

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
        suffixes=("_ultralytics", "_medical"),
    )

    merged_path = run_dir / "research_metrics_merged.csv"
    merged_df.to_csv(merged_path, index=False)

    return merged_path


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv26 segmentation on BRISC and compute train/val "
            "medical segmentation metrics per epoch."
        )
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
        "--train-csv",
        type=str,
        required=True,
        help="Path to training split CSV, e.g. data/splits/brisc_yolo_train.csv",
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        required=True,
        help="Path to validation split CSV, e.g. data/splits/brisc_yolo_val.csv",
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
        help="Confidence threshold for prediction-based medical metrics.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for prediction-based medical metrics.",
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
        "--train-eval-limit",
        type=int,
        default=0,
        help=(
            "Number of train images used for medical metric evaluation. "
            "Use 0 for full train set."
        ),
    )

    parser.add_argument(
        "--val-eval-limit",
        type=int,
        default=0,
        help=(
            "Number of validation images used for medical metric evaluation. "
            "Use 0 for full validation set."
        ),
    )

    parser.add_argument(
        "--skip-yolo-train",
        action="store_true",
        help="Skip YOLO training and only evaluate existing saved weights.",
    )

    parser.add_argument(
        "--skip-train-metrics",
        action="store_true",
        help="Only calculate validation metrics, not train metrics.",
    )

    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Save per-image train/val metric CSVs for every epoch.",
    )

    parser.add_argument(
        "--delete-epoch-weights-after-eval",
        action="store_true",
        help="Delete epoch*.pt files after medical metrics are computed. Keeps best.pt and last.pt.",
    )

    parser.add_argument(
        "--eval-weight-mode",
        type=str,
        default="epochs",
        choices=["epochs", "best", "last", "best-last", "all"],
        help=(
            "Which weights to evaluate. Use 'epochs' for per-epoch curves. "
            "Use 'best-last' for quick final comparison."
        ),
    )

    args = parser.parse_args()

    expected_run_dir = Path(args.project) / args.name

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    validate_split_csv(train_df, train_csv)
    validate_split_csv(val_df, val_csv)

    print("=" * 80)
    print("YOLOv26 BRISC RESEARCH TRAINING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data YAML: {args.data}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print(f"Expected run directory: {expected_run_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Evaluate train metrics: {not args.skip_train_metrics}")
    print(f"Train eval limit: {args.train_eval_limit}")
    print(f"Val eval limit: {args.val_eval_limit}")
    print(f"Eval weight mode: {args.eval_weight_mode}")
    print("=" * 80)

    if not args.skip_yolo_train:
        save_dir = train_yolo(args)

        if save_dir is not None and save_dir.exists():
            run_dir = save_dir
        else:
            run_dir = find_actual_run_dir(args.project, args.name)

    else:
        print("[INFO] Skipping YOLO training. Evaluating existing weights only.")
        run_dir = find_actual_run_dir(args.project, args.name)

    print(f"[INFO] Actual YOLO run directory detected: {run_dir}")

    weights = collect_weights(run_dir, mode=args.eval_weight_mode)

    print("\n" + "=" * 80)
    print("CUSTOM TRAIN + VALIDATION MEDICAL METRICS")
    print("=" * 80)
    print(f"Weights selected: {[w.name for w in weights]}")
    print("=" * 80)

    custom_rows = []
    per_sample_dir = run_dir / "per_sample_train_val_metrics"

    for weight_path in weights:
        weight_name, epoch = get_weight_name_and_epoch(weight_path)

        summary = evaluate_one_weight_on_train_and_val(
            weight_path=weight_path,
            train_df=train_df,
            val_df=val_df,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            eval_batch=args.eval_batch,
            mask_threshold=args.mask_threshold,
            train_eval_limit=args.train_eval_limit,
            val_eval_limit=args.val_eval_limit,
            save_per_sample=args.save_per_sample,
            per_sample_dir=per_sample_dir,
            evaluate_train=not args.skip_train_metrics,
        )

        custom_rows.append(summary)

        train_text = ""
        if not args.skip_train_metrics:
            train_text = (
                f"Train Dice: {summary['train_dice']:.4f} | "
                f"Train IoU: {summary['train_iou']:.4f} | "
                f"Train HD95: {summary['train_hd95']:.4f} | "
            )

        print(
            f"Epoch {epoch:03d} | "
            f"{train_text}"
            f"Val Dice: {summary['val_dice']:.4f} | "
            f"Val Dice Loss: {summary['val_dice_loss']:.4f} | "
            f"Val IoU: {summary['val_iou']:.4f} | "
            f"Val Precision: {summary['val_precision']:.4f} | "
            f"Val Recall: {summary['val_recall']:.4f} | "
            f"Val HD95: {summary['val_hd95']:.4f} | "
            f"Val ASD: {summary['val_asd']:.4f} | "
            f"Val Class Acc: {summary['val_class_accuracy']:.4f}"
        )

        if args.delete_epoch_weights_after_eval and weight_path.name.startswith("epoch"):
            try:
                os.remove(weight_path)
            except OSError:
                pass

    custom_df = pd.DataFrame(custom_rows).sort_values("epoch")

    custom_metrics_path = run_dir / "custom_train_val_metrics_by_epoch.csv"
    custom_df.to_csv(custom_metrics_path, index=False)

    merged_path = merge_with_ultralytics_results(
        run_dir=run_dir,
        custom_metrics_path=custom_metrics_path,
    )

    print("\n" + "=" * 80)
    print("RESEARCH METRICS COMPLETE")
    print("=" * 80)
    print(f"Custom train/val medical metrics saved to: {custom_metrics_path}")

    if merged_path is not None:
        print(f"Merged YOLO + medical metrics saved to: {merged_path}")

    best_row = custom_df.loc[custom_df["val_dice"].idxmax()]

    print("\nBest epoch by validation Dice:")
    print(f"Weight: {best_row['weight_name']}")
    print(f"Epoch: {int(best_row['epoch'])}")
    print(f"Val Dice: {best_row['val_dice']:.4f}")
    print(f"Val Dice Loss: {best_row['val_dice_loss']:.4f}")
    print(f"Val IoU: {best_row['val_iou']:.4f}")
    print(f"Val Precision: {best_row['val_precision']:.4f}")
    print(f"Val Recall: {best_row['val_recall']:.4f}")
    print(f"Val Specificity: {best_row['val_specificity']:.4f}")
    print(f"Val HD95: {best_row['val_hd95']:.4f}")
    print(f"Val ASD: {best_row['val_asd']:.4f}")
    print(f"Val Volume Similarity: {best_row['val_volume_similarity']:.4f}")
    print(f"Val Class Accuracy: {best_row['val_class_accuracy']:.4f}")

    if not args.skip_train_metrics:
        print("\nSame epoch training metrics:")
        print(f"Train Dice: {best_row['train_dice']:.4f}")
        print(f"Train Dice Loss: {best_row['train_dice_loss']:.4f}")
        print(f"Train IoU: {best_row['train_iou']:.4f}")
        print(f"Train Precision: {best_row['train_precision']:.4f}")
        print(f"Train Recall: {best_row['train_recall']:.4f}")
        print(f"Train Specificity: {best_row['train_specificity']:.4f}")
        print(f"Train HD95: {best_row['train_hd95']:.4f}")
        print(f"Train ASD: {best_row['train_asd']:.4f}")
        print(f"Train Volume Similarity: {best_row['train_volume_similarity']:.4f}")


if __name__ == "__main__":
    main()