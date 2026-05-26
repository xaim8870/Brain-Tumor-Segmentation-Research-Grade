# scripts/train_segformer_brisc_research.py

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation_models_pytorch is not installed. Install it using:\n"
        "pip install segmentation-models-pytorch timm"
    ) from e


# -------------------------------------------------------
# Repo imports
# -------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.metrics.segmentation_metrics import (
    STANDARD_BINARY_METRIC_COLUMNS,
    aggregate_metric_dicts,
    compute_binary_segmentation_metrics,
)


# -------------------------------------------------------
# Constants
# -------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_torch_device(device_arg: str):
    """
    Supports:
    --device 0
    --device cuda
    --device cuda:0
    --device cpu
    """

    device_arg = str(device_arg).lower()

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_arg}")
        return torch.device("cpu")

    if device_arg.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_arg)
        return torch.device("cpu")

    return torch.device("cpu")


# -------------------------------------------------------
# Image and mask loading
# -------------------------------------------------------

def read_image_rgb(image_path: Path, image_size: int):
    """
    Reads MRI image and converts it to 3-channel RGB tensor input.

    BRISC images are often grayscale, but ImageNet-pretrained SegFormer expects 3 channels.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    elif image.ndim == 3:
        channels = image.shape[2]

        if channels == 1:
            image = np.repeat(image, 3, axis=2)

        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        else:
            raise ValueError(
                f"Unsupported image channels: {channels}, shape={image.shape}, path={image_path}"
            )

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}, path={image_path}")

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image).float()


def read_mask_2d(mask_path: Path):
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
                f"Unsupported mask channels: {channels}, shape={mask.shape}, path={mask_path}"
            )

    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}, path={mask_path}")

    return mask


def read_binary_mask(mask_path: Path, image_size: int, threshold: int = 128):
    """
    Reads anti-aliased BRISC PNG mask and converts it to clean binary mask.

    Values >= threshold are tumour.
    Values < threshold are background.
    """

    mask = read_mask_2d(mask_path)

    mask = cv2.resize(
        mask,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )

    mask = (mask >= threshold).astype(np.float32)

    return torch.from_numpy(mask).unsqueeze(0).float()


def make_multiclass_mask(binary_mask: torch.Tensor, class_id: int):
    """
    Converts binary tumour mask into semantic class mask.

    For task=binary:
        target is binary mask: 0 background, 1 tumour

    For task=multiclass_type:
        target is:
            0 = background
            1 = glioma
            2 = meningioma
            3 = pituitary

    YOLO class IDs are:
        0 = glioma
        1 = meningioma
        2 = pituitary

    So multiclass tumour pixels become class_id + 1.
    """

    mask_2d = binary_mask.squeeze(0)
    output = torch.zeros_like(mask_2d, dtype=torch.long)

    tumour_class = int(class_id) + 1
    output[mask_2d > 0.5] = tumour_class

    return output.long()


def get_existing_path(row, candidate_cols):
    for col in candidate_cols:
        if col in row and pd.notna(row[col]):
            path = Path(str(row[col]))

            if path.exists():
                return path

    raise FileNotFoundError(
        f"No valid path found. Checked columns: {candidate_cols}"
    )


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

class BriscSegmentationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_size: int,
        mask_threshold: int = 128,
        limit: int = 0,
    ):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = ["mask_path", "class_id"]

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"{self.csv_path} missing required column: {col}")

        if "output_image_path" not in self.df.columns and "image_path" not in self.df.columns:
            raise ValueError(
                f"{self.csv_path} must contain output_image_path or image_path column."
            )

        if limit and limit > 0:
            self.df = self.df.head(limit).copy()

        self.image_size = image_size
        self.mask_threshold = mask_threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = get_existing_path(row, ["output_image_path", "image_path"])
        mask_path = get_existing_path(row, ["mask_path"])

        class_id = int(row["class_id"])

        image = read_image_rgb(image_path, image_size=self.image_size)
        binary_mask = read_binary_mask(
            mask_path,
            image_size=self.image_size,
            threshold=self.mask_threshold,
        )
        multiclass_mask = make_multiclass_mask(binary_mask, class_id=class_id)

        sample = {
            "image": image,
            "binary_mask": binary_mask,
            "multiclass_mask": multiclass_mask,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "tumour_label": row.get("tumour_label", "unknown"),
            "plane_label": row.get("plane_label", "unknown"),
        }

        return sample


# -------------------------------------------------------
# Losses
# -------------------------------------------------------

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (1, 2, 3)

        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice

        return loss.mean()


class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-7, ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        """
        logits: B, C, H, W
        targets: B, H, W
        """

        probs = torch.softmax(logits, dim=1)

        targets_one_hot = torch.nn.functional.one_hot(
            targets.long(),
            num_classes=self.num_classes,
        )
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        if self.ignore_background:
            probs = probs[:, 1:, :, :]
            targets_one_hot = targets_one_hot[:, 1:, :, :]

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice

        return loss.mean()


def compute_loss(
    logits,
    batch,
    task: str,
    bce_loss_fn,
    ce_loss_fn,
    binary_dice_fn,
    multiclass_dice_fn,
    bce_weight: float,
    ce_weight: float,
    dice_weight: float,
):
    if task == "binary":
        target = batch["binary_mask"].to(logits.device)

        bce_loss = bce_loss_fn(logits, target)
        dice_loss = binary_dice_fn(logits, target)

        total_loss = (bce_weight * bce_loss) + (dice_weight * dice_loss)

        return total_loss, {
            "bce_loss": float(bce_loss.detach().cpu()),
            "ce_loss": np.nan,
            "dice_loss": float(dice_loss.detach().cpu()),
        }

    if task == "multiclass_type":
        target = batch["multiclass_mask"].to(logits.device).long()

        ce_loss = ce_loss_fn(logits, target)
        dice_loss = multiclass_dice_fn(logits, target)

        total_loss = (ce_weight * ce_loss) + (dice_weight * dice_loss)

        return total_loss, {
            "bce_loss": np.nan,
            "ce_loss": float(ce_loss.detach().cpu()),
            "dice_loss": float(dice_loss.detach().cpu()),
        }

    raise ValueError(f"Unsupported task: {task}")


# -------------------------------------------------------
# Model
# -------------------------------------------------------

def create_segformer_model(
    encoder_name: str,
    encoder_weights: str | None,
    task: str,
):
    """
    task=binary:
        output channels = 1

    task=multiclass_type:
        output channels = 4
        0 background
        1 glioma
        2 meningioma
        3 pituitary
    """

    if task == "binary":
        classes = 1
    elif task == "multiclass_type":
        classes = 4
    else:
        raise ValueError(f"Unsupported task: {task}")

    if encoder_weights is not None and str(encoder_weights).lower() in ["none", "null", "false"]:
        encoder_weights = None

    if not hasattr(smp, "Segformer"):
        raise AttributeError(
            "Your installed segmentation_models_pytorch does not have smp.Segformer. "
            "Upgrade it using: pip install -U segmentation-models-pytorch timm"
        )

    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
        activation=None,
    )

    return model


# -------------------------------------------------------
# Train and evaluation loops
# -------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    task,
    bce_loss_fn,
    ce_loss_fn,
    binary_dice_fn,
    multiclass_dice_fn,
    bce_weight,
    ce_weight,
    dice_weight,
    use_amp,
):
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)

            loss, components = compute_loss(
                logits=logits,
                batch=batch,
                task=task,
                bce_loss_fn=bce_loss_fn,
                ce_loss_fn=ce_loss_fn,
                binary_dice_fn=binary_dice_fn,
                multiclass_dice_fn=multiclass_dice_fn,
                bce_weight=bce_weight,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_size

        if not np.isnan(components["bce_loss"]):
            total_bce += components["bce_loss"] * batch_size

        if not np.isnan(components["ce_loss"]):
            total_ce += components["ce_loss"] * batch_size

        total_dice += components["dice_loss"] * batch_size
        total_samples += batch_size

        pbar.set_postfix(loss=float(loss.detach().cpu()))

    avg_loss = total_loss / max(total_samples, 1)
    avg_bce = total_bce / max(total_samples, 1) if total_bce > 0 else np.nan
    avg_ce = total_ce / max(total_samples, 1) if total_ce > 0 else np.nan
    avg_dice = total_dice / max(total_samples, 1)

    return {
        "train_loss": avg_loss,
        "train_bce_loss": avg_bce,
        "train_ce_loss": avg_ce,
        "train_dice_loss": avg_dice,
    }


@torch.no_grad()
def evaluate_split(
    model,
    loader,
    device,
    task,
    split_name,
    bce_loss_fn,
    ce_loss_fn,
    binary_dice_fn,
    multiclass_dice_fn,
    bce_weight,
    ce_weight,
    dice_weight,
    pred_threshold,
):
    model.eval()

    metric_rows = []

    total_loss = 0.0
    total_bce = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Evaluating {split_name}", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)

        loss, components = compute_loss(
            logits=logits,
            batch=batch,
            task=task,
            bce_loss_fn=bce_loss_fn,
            ce_loss_fn=ce_loss_fn,
            binary_dice_fn=binary_dice_fn,
            multiclass_dice_fn=multiclass_dice_fn,
            bce_weight=bce_weight,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )

        total_loss += float(loss.detach().cpu()) * batch_size

        if not np.isnan(components["bce_loss"]):
            total_bce += components["bce_loss"] * batch_size

        if not np.isnan(components["ce_loss"]):
            total_ce += components["ce_loss"] * batch_size

        total_dice += components["dice_loss"] * batch_size
        total_samples += batch_size

        if task == "binary":
            probs = torch.sigmoid(logits)
            preds = probs >= pred_threshold

            gt_binary = batch["binary_mask"].bool()

            for idx in range(batch_size):
                pred_mask = preds[idx, 0].detach().cpu().numpy().astype(bool)
                gt_mask = gt_binary[idx, 0].detach().cpu().numpy().astype(bool)

                metrics = compute_binary_segmentation_metrics(
                    pred=pred_mask,
                    gt=gt_mask,
                    pred_threshold=None,
                    gt_threshold=None,
                    spacing=None,
                )

                metrics["tumour_label"] = batch["tumour_label"][idx]
                metrics["plane_label"] = batch["plane_label"][idx]
                metric_rows.append(metrics)

        elif task == "multiclass_type":
            pred_class_map = torch.argmax(logits, dim=1)
            pred_binary = pred_class_map > 0

            gt_binary = batch["binary_mask"].bool()

            for idx in range(batch_size):
                pred_mask = pred_binary[idx].detach().cpu().numpy().astype(bool)
                gt_mask = gt_binary[idx, 0].detach().cpu().numpy().astype(bool)

                metrics = compute_binary_segmentation_metrics(
                    pred=pred_mask,
                    gt=gt_mask,
                    pred_threshold=None,
                    gt_threshold=None,
                    spacing=None,
                )

                metrics["tumour_label"] = batch["tumour_label"][idx]
                metrics["plane_label"] = batch["plane_label"][idx]
                metric_rows.append(metrics)

    metric_dicts = [
        {k: v for k, v in row.items() if isinstance(v, (int, float, np.integer, np.floating))}
        for row in metric_rows
    ]

    summary = aggregate_metric_dicts(metric_dicts, prefix=f"{split_name}_")

    summary[f"{split_name}_loss"] = total_loss / max(total_samples, 1)
    summary[f"{split_name}_bce_loss"] = total_bce / max(total_samples, 1) if total_bce > 0 else np.nan
    summary[f"{split_name}_ce_loss"] = total_ce / max(total_samples, 1) if total_ce > 0 else np.nan
    summary[f"{split_name}_dice_loss_from_loss_fn"] = total_dice / max(total_samples, 1)

    # Class-wise binary segmentation metrics
    sample_df = pd.DataFrame(metric_rows)

    if not sample_df.empty:
        for tumour_label, group in sample_df.groupby("tumour_label"):
            clean_label = str(tumour_label).strip().lower().replace(" ", "_")

            summary[f"{split_name}_{clean_label}_dice"] = float(group["dice"].mean())
            summary[f"{split_name}_{clean_label}_iou"] = float(group["iou"].mean())
            summary[f"{split_name}_{clean_label}_hd95"] = float(group["hd95"].mean())
            summary[f"{split_name}_{clean_label}_asd"] = float(group["asd"].mean())

    return summary


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_dice: float, args):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }

    torch.save(checkpoint, path)


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def append_metrics_csv(metrics_path: Path, row: Dict):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row])

    if metrics_path.exists():
        old_df = pd.read_csv(metrics_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df

    new_df.to_csv(metrics_path, index=False)


def save_args(out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train SegFormer on BRISC and save train/val medical metrics per epoch."
    )

    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)

    parser.add_argument("--out-dir", type=str, default="experiments/brisc_segformer")
    parser.add_argument("--run-name", type=str, default="segformer_mitb0_binary")

    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "multiclass_type"],
        help=(
            "binary = tumour vs background. "
            "multiclass_type = background/glioma/meningioma/pituitary semantic segmentation."
        ),
    )

    parser.add_argument("--encoder", type=str, default="mit_b0")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)

    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--pred-threshold", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--train-eval-limit",
        type=int,
        default=0,
        help="Use 0 for full train metric evaluation. Use small value for faster experiments.",
    )

    parser.add_argument(
        "--val-eval-limit",
        type=int,
        default=0,
        help="Use 0 for full validation metric evaluation.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = get_torch_device(args.device)

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics_by_epoch.csv"
    best_ckpt_path = out_dir / "checkpoints" / "best_model.pth"
    last_ckpt_path = out_dir / "checkpoints" / "last_model.pth"

    save_args(out_dir, args)

    print("=" * 80)
    print("BRISC SEGFORMER RESEARCH TRAINING")
    print("=" * 80)
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Output directory: {out_dir}")
    print(f"Task: {args.task}")
    print(f"Encoder: {args.encoder}")
    print(f"Encoder weights: {args.encoder_weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {device}")
    print(f"AMP: {args.amp}")
    print("=" * 80)

    train_dataset = BriscSegmentationDataset(
        csv_path=args.train_csv,
        image_size=args.imgsz,
        mask_threshold=args.mask_threshold,
        limit=0,
    )

    val_dataset = BriscSegmentationDataset(
        csv_path=args.val_csv,
        image_size=args.imgsz,
        mask_threshold=args.mask_threshold,
        limit=0,
    )

    train_eval_dataset = BriscSegmentationDataset(
        csv_path=args.train_csv,
        image_size=args.imgsz,
        mask_threshold=args.mask_threshold,
        limit=args.train_eval_limit,
    )

    val_eval_dataset = BriscSegmentationDataset(
        csv_path=args.val_csv,
        image_size=args.imgsz,
        mask_threshold=args.mask_threshold,
        limit=args.val_eval_limit,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_eval_loader = DataLoader(
        val_eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    encoder_weights = args.encoder_weights

    if encoder_weights.lower() in ["none", "null", "false"]:
        encoder_weights = None

    model = create_segformer_model(
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        task=args.task,
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    binary_dice_fn = BinaryDiceLoss()
    multiclass_dice_fn = MulticlassDiceLoss(num_classes=4, ignore_background=True)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            task=args.task,
            bce_loss_fn=bce_loss_fn,
            ce_loss_fn=ce_loss_fn,
            binary_dice_fn=binary_dice_fn,
            multiclass_dice_fn=multiclass_dice_fn,
            bce_weight=args.bce_weight,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            use_amp=use_amp,
        )

        train_metrics = evaluate_split(
            model=model,
            loader=train_eval_loader,
            device=device,
            task=args.task,
            split_name="train",
            bce_loss_fn=bce_loss_fn,
            ce_loss_fn=ce_loss_fn,
            binary_dice_fn=binary_dice_fn,
            multiclass_dice_fn=multiclass_dice_fn,
            bce_weight=args.bce_weight,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            pred_threshold=args.pred_threshold,
        )

        val_metrics = evaluate_split(
            model=model,
            loader=val_eval_loader,
            device=device,
            task=args.task,
            split_name="val",
            bce_loss_fn=bce_loss_fn,
            ce_loss_fn=ce_loss_fn,
            binary_dice_fn=binary_dice_fn,
            multiclass_dice_fn=multiclass_dice_fn,
            bce_weight=args.bce_weight,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            pred_threshold=args.pred_threshold,
        )

        scheduler.step()

        epoch_time = time.time() - start_time
        lr = get_current_lr(optimizer)

        row = {
            "epoch": epoch,
            "time_sec": epoch_time,
            "lr": lr,
            **train_losses,
            **train_metrics,
            **val_metrics,
        }

        append_metrics_csv(metrics_path, row)

        val_dice = row.get("val_dice", 0.0)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                args=args,
            )

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            args=args,
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] | "
            f"Train Loss: {row['train_loss']:.4f} | "
            f"Train Dice: {row['train_dice']:.4f} | "
            f"Train IoU: {row['train_iou']:.4f} | "
            f"Train HD95: {row['train_hd95']:.4f} | "
            f"Val Loss: {row['val_loss']:.4f} | "
            f"Val Dice: {row['val_dice']:.4f} | "
            f"Val IoU: {row['val_iou']:.4f} | "
            f"Val HD95: {row['val_hd95']:.4f} | "
            f"Val ASD: {row['val_asd']:.4f} | "
            f"Best Val Dice: {best_val_dice:.4f} @ Epoch {best_epoch}"
        )

    print("\n" + "=" * 80)
    print("SEGFORMER TRAINING COMPLETE")
    print("=" * 80)
    print(f"Metrics CSV saved to: {metrics_path}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Last checkpoint saved to: {last_ckpt_path}")
    print(f"Best Val Dice: {best_val_dice:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()