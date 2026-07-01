# scripts/train_segformer_brisc_research.py

from pathlib import Path
import argparse
import gc
import random
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation, SegformerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.metrics.segmentation_metrics import (
    STANDARD_BINARY_METRIC_COLUMNS,
    compute_binary_segmentation_metrics,
)


# -----------------------------
# Reproducibility
# -----------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Path helpers
# -----------------------------

def get_image_path(row):
    for col in ["output_image_path", "image_path"]:
        if col in row and pd.notna(row[col]):
            p = Path(str(row[col]))
            if p.exists():
                return p
    raise FileNotFoundError("No valid image path found.")


def get_mask_path(row):
    if "mask_path" not in row or pd.isna(row["mask_path"]):
        raise FileNotFoundError("mask_path missing.")

    p = Path(str(row["mask_path"]))
    if not p.exists():
        raise FileNotFoundError(f"Mask path does not exist: {p}")

    return p


def read_image_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_mask_binary(mask_path: Path, threshold=128):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    if mask.ndim == 3:
        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = mask[:, :, 0]

    return (mask >= threshold).astype(np.uint8)


# -----------------------------
# Augmentation
# -----------------------------

def random_affine(img, mask, degrees=5, scale_range=(0.9, 1.1), translate=0.03):
    h, w = img.shape[:2]

    angle = random.uniform(-degrees, degrees)
    scale = random.uniform(scale_range[0], scale_range[1])

    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h

    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[:, 2] += [tx, ty]

    img_aug = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask_aug = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img_aug, mask_aug


def augment_train(img, mask):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.7:
        img, mask = random_affine(
            img,
            mask,
            degrees=5,
            scale_range=(0.9, 1.1),
            translate=0.03,
        )

    if random.random() < 0.3:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-8, 8)
        img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.15:
        noise = np.random.normal(0, 4, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img, mask


# -----------------------------
# Dataset
# -----------------------------

class BriscSegDataset(Dataset):
    def __init__(self, csv_path, imgsz=512, mask_threshold=128, train=False):
        self.df = pd.read_csv(csv_path)
        self.imgsz = imgsz
        self.mask_threshold = mask_threshold
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = get_image_path(row)
        mask_path = get_mask_path(row)

        img = read_image_rgb(img_path)
        mask = read_mask_binary(mask_path, threshold=self.mask_threshold)

        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.train:
            img, mask = augment_train(img, mask)

        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image_filename": img_path.name,
            "mask_filename": mask_path.name,
            "tumour_label": row.get("tumour_label", "unknown"),
            "plane_label": row.get("plane_label", "unknown"),
            "class_id": int(row["class_id"]) if "class_id" in row and pd.notna(row["class_id"]) else -1,
        }


# -----------------------------
# Loss
# -----------------------------

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.7, ce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.register_buffer("ce_weight", torch.tensor([0.2, 0.8], dtype=torch.float32))

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, weight=self.ce_weight.to(logits.device))

        probs = torch.softmax(logits, dim=1)[:, 1]
        target_fg = (target == 1).float()

        intersection = (probs * target_fg).sum(dim=(1, 2))
        denominator = probs.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))

        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice.mean()

        return (self.dice_weight * dice_loss) + (self.ce_weight * ce_loss)


# -----------------------------
# Postprocessing
# -----------------------------

def postprocess_pred_mask(pred_mask, min_component_area=30, keep_largest=True, fill_holes=True):
    mask = pred_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask.astype(bool)

    components = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            components.append((label_id, area))

    if not components:
        return np.zeros_like(mask).astype(bool)

    clean = np.zeros_like(mask)

    if keep_largest:
        largest_id = max(components, key=lambda x: x[1])[0]
        clean[labels == largest_id] = 1
    else:
        for label_id, _ in components:
            clean[labels == label_id] = 1

    if fill_holes:
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(clean)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        clean = filled

    return clean.astype(bool)


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def evaluate_segformer(
    model,
    loader,
    device,
    epoch,
    split_name,
    threshold=0.5,
    save_per_sample=False,
    per_sample_dir=None,
    postprocess=True,
):
    model.eval()
    sample_rows = []

    for batch in tqdm(loader, desc=f"Evaluating {split_name} epoch {epoch}", leave=False):
        images = batch["image"].to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).cpu().numpy().astype(np.uint8)

        gts = batch["mask"].cpu().numpy().astype(np.uint8)

        for i in range(len(preds)):
            pred_mask = preds[i]
            gt_mask = gts[i].astype(bool)

            if postprocess:
                pred_mask = postprocess_pred_mask(
                    pred_mask,
                    min_component_area=30,
                    keep_largest=True,
                    fill_holes=True,
                )
            else:
                pred_mask = pred_mask.astype(bool)

            metrics = compute_binary_segmentation_metrics(
                pred=pred_mask,
                gt=gt_mask,
                pred_threshold=None,
                gt_threshold=None,
                spacing=None,
            )

            sample_rows.append({
                "epoch": epoch,
                "split": split_name,
                "image_filename": batch["image_filename"][i],
                "mask_filename": batch["mask_filename"][i],
                "tumour_label": batch["tumour_label"][i],
                "plane_label": batch["plane_label"][i],
                "gt_class_id": int(batch["class_id"][i]),
                **metrics,
            })

    sample_df = pd.DataFrame(sample_rows)

    if save_per_sample and per_sample_dir is not None:
        out_dir = per_sample_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(out_dir / f"{split_name}_sample_metrics_epoch_{epoch:03d}.csv", index=False)

    summary = {
        f"{split_name}_num_images": int(len(sample_df)),
    }

    for col in STANDARD_BINARY_METRIC_COLUMNS:
        if col in sample_df.columns:
            summary[f"{split_name}_{col}"] = float(sample_df[col].mean())

    for tumour_label, group in sample_df.groupby("tumour_label"):
        clean_label = str(tumour_label).strip().lower().replace(" ", "_")
        summary[f"{split_name}_{clean_label}_dice"] = float(group["dice"].mean())
        summary[f"{split_name}_{clean_label}_iou"] = float(group["iou"].mean())
        summary[f"{split_name}_{clean_label}_hd95"] = float(group["hd95"].mean())
        summary[f"{split_name}_{clean_label}_asd"] = float(group["asd"].mean())

    return summary


# -----------------------------
# Model
# -----------------------------

def build_model(args):
    if args.from_scratch:
        config = SegformerConfig(
            num_labels=2,
            id2label={0: "background", 1: "tumour"},
            label2id={"background": 0, "tumour": 1},
        )
        model = SegformerForSemanticSegmentation(config)
        return model

    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
            id2label={0: "background", 1: "tumour"},
            label2id={"background": 0, "tumour": 1},
        )
        return model

    except Exception as e:
        print(f"[WARNING] Could not load pretrained model: {args.model_name}")
        print(f"[WARNING] Error: {e}")
        print("[INFO] Falling back to SegFormer from scratch.")

        config = SegformerConfig(
            num_labels=2,
            id2label={0: "background", 1: "tumour"},
            label2id={"background": 0, "tumour": 1},
        )
        model = SegformerForSemanticSegmentation(config)
        return model


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=images)
                logits = outputs.logits

                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(pixel_values=images)
            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def compute_val_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Validation loss", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def save_checkpoint(path, model, optimizer, epoch, best_metric):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }, path)


def merge_results(results_path, custom_metrics_path, out_path):
    results_df = pd.read_csv(results_path)
    custom_df = pd.read_csv(custom_metrics_path)

    merged = results_df.merge(
        custom_df,
        on="epoch",
        how="left",
        suffixes=("_training", "_medical"),
    )

    merged.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--from-scratch", action="store_true")

    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)

    parser.add_argument("--project", type=str, default="experiments/brisc_segformer")
    parser.add_argument("--name", type=str, default="SegFormer-B2-Clean-512")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--pred-threshold", type=float, default=0.5)

    parser.add_argument("--train-eval-limit", type=int, default=300)
    parser.add_argument("--val-eval-limit", type=int, default=0)

    parser.add_argument("--save-per-sample", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)

    args = parser.parse_args()

    seed_everything(args.seed)

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    results_path = run_dir / "results.csv"
    custom_metrics_path = run_dir / "custom_train_val_metrics_by_epoch.csv"
    merged_path = run_dir / "research_metrics_merged.csv"
    per_sample_dir = run_dir / "per_sample_train_val_metrics"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = BriscSegDataset(
        args.train_csv,
        imgsz=args.imgsz,
        mask_threshold=args.mask_threshold,
        train=True,
    )

    val_dataset = BriscSegDataset(
        args.val_csv,
        imgsz=args.imgsz,
        mask_threshold=args.mask_threshold,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.train_eval_limit and args.train_eval_limit > 0:
        train_eval_df = pd.read_csv(args.train_csv).head(args.train_eval_limit)
        train_eval_csv = run_dir / "train_eval_subset.csv"
        train_eval_df.to_csv(train_eval_csv, index=False)
        train_eval_dataset = BriscSegDataset(
            train_eval_csv,
            imgsz=args.imgsz,
            mask_threshold=args.mask_threshold,
            train=False,
        )
    else:
        train_eval_dataset = BriscSegDataset(
            args.train_csv,
            imgsz=args.imgsz,
            mask_threshold=args.mask_threshold,
            train=False,
        )

    if args.val_eval_limit and args.val_eval_limit > 0:
        val_eval_df = pd.read_csv(args.val_csv).head(args.val_eval_limit)
        val_eval_csv = run_dir / "val_eval_subset.csv"
        val_eval_df.to_csv(val_eval_csv, index=False)
        val_eval_dataset = BriscSegDataset(
            val_eval_csv,
            imgsz=args.imgsz,
            mask_threshold=args.mask_threshold,
            train=False,
        )
    else:
        val_eval_dataset = val_dataset

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_eval_loader = DataLoader(
        val_eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = build_model(args).to(device)

    criterion = DiceCELoss(dice_weight=0.7, ce_weight=0.3)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def poly_lr(epoch):
        return (1 - epoch / args.epochs) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_val_dice = -1.0
    best_epoch = -1
    bad_epochs = 0

    results_rows = []
    custom_rows = []

    print("=" * 80)
    print("SEGFORMER BRISC RESEARCH TRAINING")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Run dir: {run_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Device: {device}")
    print(f"LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )

        val_loss = compute_val_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        result_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        }

        results_rows.append(result_row)
        pd.DataFrame(results_rows).to_csv(results_path, index=False)

        train_summary = evaluate_segformer(
            model=model,
            loader=train_eval_loader,
            device=device,
            epoch=epoch,
            split_name="train",
            threshold=args.pred_threshold,
            save_per_sample=False,
            per_sample_dir=per_sample_dir,
            postprocess=True,
        )

        val_summary = evaluate_segformer(
            model=model,
            loader=val_eval_loader,
            device=device,
            epoch=epoch,
            split_name="val",
            threshold=args.pred_threshold,
            save_per_sample=args.save_per_sample,
            per_sample_dir=per_sample_dir,
            postprocess=True,
        )

        custom_row = {
            "epoch": epoch,
            "weight_name": f"epoch_{epoch:03d}",
            "weight_path": str(weights_dir / f"epoch{epoch}.pt"),
            **train_summary,
            **val_summary,
        }

        custom_rows.append(custom_row)
        pd.DataFrame(custom_rows).to_csv(custom_metrics_path, index=False)

        merge_results(results_path, custom_metrics_path, merged_path)

        save_checkpoint(
            weights_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_dice,
        )

        save_checkpoint(
            weights_dir / f"epoch{epoch}.pt",
            model,
            optimizer,
            epoch,
            best_val_dice,
        )

        val_dice = val_summary["val_dice"]

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint(
                weights_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_dice,
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Dice: {train_summary['train_dice']:.4f} | "
            f"Train IoU: {train_summary['train_iou']:.4f} | "
            f"Val Dice: {val_summary['val_dice']:.4f} | "
            f"Val IoU: {val_summary['val_iou']:.4f} | "
            f"Val Precision: {val_summary['val_precision']:.4f} | "
            f"Val Recall: {val_summary['val_recall']:.4f} | "
            f"Val HD95: {val_summary['val_hd95']:.4f} | "
            f"Val ASD: {val_summary['val_asd']:.4f}"
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if bad_epochs >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    custom_df = pd.read_csv(custom_metrics_path)
    best_row = custom_df.loc[custom_df["val_dice"].idxmax()]

    print("\n" + "=" * 80)
    print("SEGFORMER RESEARCH METRICS COMPLETE")
    print("=" * 80)
    print(f"Results CSV: {results_path}")
    print(f"Custom metrics CSV: {custom_metrics_path}")
    print(f"Merged metrics CSV: {merged_path}")

    print("\nBest epoch by validation Dice:")
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


if __name__ == "__main__":
    main()