from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


EPS = 1e-7

ArrayLike = Union[np.ndarray, torch.Tensor]


# -------------------------------------------------------
# Conversion utilities
# -------------------------------------------------------

def to_numpy(array: ArrayLike) -> np.ndarray:
    """
    Converts torch.Tensor or numpy.ndarray to numpy.ndarray.

    Supports:
    - torch tensor on CPU/GPU
    - numpy array
    """

    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()

    if isinstance(array, np.ndarray):
        return array

    raise TypeError(f"Unsupported array type: {type(array)}")


def squeeze_mask(mask: ArrayLike) -> np.ndarray:
    """
    Converts a mask to a 2D or 3D numpy array.

    Common supported shapes:
    - H, W
    - 1, H, W
    - H, W, 1
    - B, H, W
    - B, 1, H, W

    This function does not threshold. It only removes unnecessary singleton channels.
    """

    mask = to_numpy(mask)

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        # H, W, 1
        if mask.shape[-1] == 1:
            return mask[:, :, 0]

        # 1, H, W
        if mask.shape[0] == 1:
            return mask[0]

        # B, H, W
        return mask

    if mask.ndim == 4:
        # B, 1, H, W
        if mask.shape[1] == 1:
            return mask[:, 0]

        # B, H, W, 1
        if mask.shape[-1] == 1:
            return mask[:, :, :, 0]

    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def binarize_mask(mask: ArrayLike, threshold: float = 0.5) -> np.ndarray:
    """
    Converts a mask/probability map into a boolean binary mask.

    For model probabilities/logits after sigmoid:
        threshold = 0.5

    For PNG masks with 0-255 values:
        threshold = 128
    """

    mask = squeeze_mask(mask)

    return mask >= threshold


def ensure_same_shape(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch. pred={pred.shape}, gt={gt.shape}")


# -------------------------------------------------------
# Confusion counts
# -------------------------------------------------------

def binary_confusion_counts(pred: ArrayLike, gt: ArrayLike) -> Dict[str, int]:
    """
    Computes pixel-level TP, FP, FN, TN for binary segmentation.

    Inputs must already be binary/boolean masks with same shape.
    """

    pred = squeeze_mask(pred).astype(bool)
    gt = squeeze_mask(gt).astype(bool)

    ensure_same_shape(pred, gt)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# -------------------------------------------------------
# Overlap metrics
# -------------------------------------------------------

def dice_score_from_counts(tp: int, fp: int, fn: int, zero_division: float = 1.0) -> float:
    denominator = (2 * tp) + fp + fn

    if denominator == 0:
        return float(zero_division)

    return float((2 * tp) / (denominator + EPS))


def iou_score_from_counts(tp: int, fp: int, fn: int, zero_division: float = 1.0) -> float:
    denominator = tp + fp + fn

    if denominator == 0:
        return float(zero_division)

    return float(tp / (denominator + EPS))


def precision_from_counts(tp: int, fp: int, zero_division: float = 0.0) -> float:
    denominator = tp + fp

    if denominator == 0:
        return float(zero_division)

    return float(tp / (denominator + EPS))


def recall_from_counts(tp: int, fn: int, zero_division: float = 0.0) -> float:
    denominator = tp + fn

    if denominator == 0:
        return float(zero_division)

    return float(tp / (denominator + EPS))


def specificity_from_counts(tn: int, fp: int, zero_division: float = 1.0) -> float:
    denominator = tn + fp

    if denominator == 0:
        return float(zero_division)

    return float(tn / (denominator + EPS))


def volume_similarity_from_masks(pred: ArrayLike, gt: ArrayLike, zero_division: float = 1.0) -> float:
    """
    Volume similarity compares predicted tumour size with ground-truth tumour size.

    Range:
    - 1.0 = same volume
    - lower value = larger volume mismatch
    """

    pred = squeeze_mask(pred).astype(bool)
    gt = squeeze_mask(gt).astype(bool)

    ensure_same_shape(pred, gt)

    pred_volume = float(pred.sum())
    gt_volume = float(gt.sum())

    denominator = pred_volume + gt_volume

    if denominator == 0:
        return float(zero_division)

    return float(1.0 - (abs(pred_volume - gt_volume) / (denominator + EPS)))


def compute_overlap_metrics(
    pred: ArrayLike,
    gt: ArrayLike,
    zero_division: float = 1.0,
) -> Dict[str, float]:
    """
    Computes standard pixel-overlap segmentation metrics.

    If pred and gt are both empty:
    - Dice = 1
    - IoU = 1
    - Volume similarity = 1

    This is useful for future no-tumour empty-mask samples.
    """

    pred = squeeze_mask(pred).astype(bool)
    gt = squeeze_mask(gt).astype(bool)

    ensure_same_shape(pred, gt)

    counts = binary_confusion_counts(pred, gt)

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    dice = dice_score_from_counts(tp, fp, fn, zero_division=zero_division)
    iou = iou_score_from_counts(tp, fp, fn, zero_division=zero_division)
    precision = precision_from_counts(tp, fp, zero_division=0.0)
    recall = recall_from_counts(tp, fn, zero_division=0.0)
    specificity = specificity_from_counts(tn, fp, zero_division=1.0)
    volume_similarity = volume_similarity_from_masks(pred, gt, zero_division=zero_division)

    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())

    return {
        "dice": float(dice),
        "dice_loss": float(1.0 - dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "volume_similarity": float(volume_similarity),
        "pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# -------------------------------------------------------
# Boundary metrics
# -------------------------------------------------------

def get_surface(mask: ArrayLike) -> np.ndarray:
    """
    Extracts the binary surface/boundary of a mask.
    """

    mask = squeeze_mask(mask).astype(bool)

    if not mask.any():
        return mask

    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )

    surface = np.logical_xor(mask, eroded)

    if not surface.any():
        return mask

    return surface


def _normalise_spacing(spacing: Optional[Sequence[float]], ndim: int) -> Tuple[float, ...]:
    """
    Spacing is useful for BraTS NIfTI data.

    For BRISC PNG/JPG data:
        spacing=None
        distances are measured in pixels.

    For BraTS:
        spacing=(sx, sy) for 2D
        spacing=(sx, sy, sz) for 3D
    """

    if spacing is None:
        return tuple([1.0] * ndim)

    if len(spacing) != ndim:
        raise ValueError(f"spacing length must match mask ndim. spacing={spacing}, ndim={ndim}")

    return tuple(float(s) for s in spacing)


def surface_distances(
    pred: ArrayLike,
    gt: ArrayLike,
    spacing: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Computes symmetric surface distances between prediction and ground truth.

    Returns distances from:
    - prediction surface to ground-truth surface
    - ground-truth surface to prediction surface

    If one mask is empty and the other is not, returns the image diagonal.
    """

    pred = squeeze_mask(pred).astype(bool)
    gt = squeeze_mask(gt).astype(bool)

    ensure_same_shape(pred, gt)

    ndim = gt.ndim
    spacing_tuple = _normalise_spacing(spacing, ndim=ndim)

    image_shape = np.array(gt.shape, dtype=np.float32)
    spacing_arr = np.array(spacing_tuple, dtype=np.float32)
    max_distance = float(np.linalg.norm(image_shape * spacing_arr))

    if not pred.any() and not gt.any():
        return np.array([0.0], dtype=np.float32)

    if pred.any() != gt.any():
        return np.array([max_distance], dtype=np.float32)

    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    dt_gt = distance_transform_edt(np.logical_not(gt_surface), sampling=spacing_tuple)
    dt_pred = distance_transform_edt(np.logical_not(pred_surface), sampling=spacing_tuple)

    pred_to_gt = dt_gt[pred_surface]
    gt_to_pred = dt_pred[gt_surface]

    distances = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float32)

    if distances.size == 0:
        return np.array([0.0], dtype=np.float32)

    return distances


def hausdorff_distance(
    pred: ArrayLike,
    gt: ArrayLike,
    spacing: Optional[Sequence[float]] = None,
) -> float:
    distances = surface_distances(pred, gt, spacing=spacing)
    return float(np.max(distances))


def hd95(
    pred: ArrayLike,
    gt: ArrayLike,
    spacing: Optional[Sequence[float]] = None,
) -> float:
    distances = surface_distances(pred, gt, spacing=spacing)
    return float(np.percentile(distances, 95))


def average_surface_distance(
    pred: ArrayLike,
    gt: ArrayLike,
    spacing: Optional[Sequence[float]] = None,
) -> float:
    distances = surface_distances(pred, gt, spacing=spacing)
    return float(np.mean(distances))


def compute_boundary_metrics(
    pred: ArrayLike,
    gt: ArrayLike,
    spacing: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Computes HD, HD95 and ASD.

    BRISC:
        spacing=None, values are in pixels.

    BraTS:
        provide voxel spacing if available, values can be in millimetres.
    """

    distances = surface_distances(pred, gt, spacing=spacing)

    return {
        "hd": float(np.max(distances)),
        "hd95": float(np.percentile(distances, 95)),
        "asd": float(np.mean(distances)),
    }


# -------------------------------------------------------
# Complete binary metric package
# -------------------------------------------------------

def compute_binary_segmentation_metrics(
    pred: ArrayLike,
    gt: ArrayLike,
    pred_threshold: Optional[float] = None,
    gt_threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Computes all medical segmentation metrics for one binary mask pair.

    Parameters
    ----------
    pred:
        Predicted mask or probability map.
    gt:
        Ground-truth mask.
    pred_threshold:
        If not None, pred is thresholded using this value.
        Use 0.5 for sigmoid probabilities.
    gt_threshold:
        If not None, gt is thresholded using this value.
        Use 128 for PNG masks.
    spacing:
        Optional physical spacing. For BRISC keep None. For BraTS use voxel spacing.

    Returns
    -------
    Dictionary containing:
    - Dice
    - Dice Loss
    - IoU
    - Precision
    - Recall / Sensitivity
    - Specificity
    - Volume Similarity
    - HD
    - HD95
    - ASD
    - TP, FP, FN, TN
    """

    if pred_threshold is not None:
        pred = binarize_mask(pred, threshold=pred_threshold)
    else:
        pred = squeeze_mask(pred).astype(bool)

    if gt_threshold is not None:
        gt = binarize_mask(gt, threshold=gt_threshold)
    else:
        gt = squeeze_mask(gt).astype(bool)

    ensure_same_shape(pred, gt)

    metrics = {}
    metrics.update(compute_overlap_metrics(pred, gt))
    metrics.update(compute_boundary_metrics(pred, gt, spacing=spacing))

    return metrics


# -------------------------------------------------------
# Batch aggregation
# -------------------------------------------------------

def aggregate_metric_dicts(
    metric_dicts: List[Dict[str, float]],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Aggregates a list of per-sample metric dictionaries by mean.

    This is what we should use for validation/test CSV rows.
    """

    if len(metric_dicts) == 0:
        raise ValueError("metric_dicts is empty. Cannot aggregate metrics.")

    keys = sorted(metric_dicts[0].keys())
    summary = {}

    for key in keys:
        values = []

        for item in metric_dicts:
            value = item.get(key, np.nan)

            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))

        if values:
            summary[f"{prefix}{key}"] = float(np.nanmean(values))

    summary[f"{prefix}num_samples"] = int(len(metric_dicts))

    return summary


def compute_batch_binary_segmentation_metrics(
    preds: Iterable[ArrayLike],
    gts: Iterable[ArrayLike],
    pred_threshold: Optional[float] = None,
    gt_threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Computes metrics for many masks and returns averaged metrics.
    """

    metric_dicts = []

    for pred, gt in zip(preds, gts):
        metrics = compute_binary_segmentation_metrics(
            pred=pred,
            gt=gt,
            pred_threshold=pred_threshold,
            gt_threshold=gt_threshold,
            spacing=spacing,
        )
        metric_dicts.append(metrics)

    return aggregate_metric_dicts(metric_dicts, prefix=prefix)


# -------------------------------------------------------
# Multiclass metrics
# -------------------------------------------------------

def compute_multiclass_segmentation_metrics(
    pred: ArrayLike,
    gt: ArrayLike,
    class_ids: Sequence[int],
    ignore_background: bool = True,
    spacing: Optional[Sequence[float]] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Computes one-vs-rest segmentation metrics for multiclass masks.

    Useful for BraTS multiclass segmentation.

    Example class_ids for BraTS:
        [0, 1, 2, 4]

    If ignore_background=True:
        class 0 is skipped.
    """

    pred = squeeze_mask(pred)
    gt = squeeze_mask(gt)

    ensure_same_shape(pred, gt)

    output = {}
    per_class_metrics = []

    for class_id in class_ids:
        if ignore_background and class_id == 0:
            continue

        pred_c = pred == class_id
        gt_c = gt == class_id

        metrics = compute_binary_segmentation_metrics(
            pred=pred_c,
            gt=gt_c,
            spacing=spacing,
        )

        class_prefix = f"{prefix}class_{class_id}_"

        for key, value in metrics.items():
            output[f"{class_prefix}{key}"] = value

        per_class_metrics.append(metrics)

    if per_class_metrics:
        output.update(aggregate_metric_dicts(per_class_metrics, prefix=f"{prefix}macro_"))

    return output


# -------------------------------------------------------
# Standard columns for CSV files
# -------------------------------------------------------

STANDARD_BINARY_METRIC_COLUMNS = [
    "dice",
    "dice_loss",
    "iou",
    "precision",
    "recall",
    "sensitivity",
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


def get_standard_binary_metric_columns(prefix: str = "") -> List[str]:
    return [f"{prefix}{col}" for col in STANDARD_BINARY_METRIC_COLUMNS]