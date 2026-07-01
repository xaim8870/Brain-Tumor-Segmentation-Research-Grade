from pathlib import Path
import argparse, gc, random, sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.metrics.segmentation_metrics import (
    STANDARD_BINARY_METRIC_COLUMNS,
    compute_binary_segmentation_metrics,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask_binary(path, threshold=1):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask >= threshold).astype(np.uint8)


def augment_train(img, mask):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.7:
        h, w = img.shape[:2]
        angle = random.uniform(-7, 7)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-0.03, 0.03) * w
        ty = random.uniform(-0.03, 0.03) * h

        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[:, 2] += [tx, ty]

        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if random.random() < 0.3:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-8, 8)
        img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.15:
        noise = np.random.normal(0, 4, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img, mask


class BriscBinaryDataset(Dataset):
    def __init__(self, csv_path, imgsz=512, mask_threshold=1, train=False):
        self.df = pd.read_csv(csv_path)
        self.imgsz = imgsz
        self.mask_threshold = mask_threshold
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        img = read_image_rgb(img_path)
        mask = read_mask_binary(mask_path, self.mask_threshold)

        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.train:
            img, mask = augment_train(img, mask)

        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img.transpose(2, 0, 1)

        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "mask": torch.tensor(mask[None, :, :], dtype=torch.float32),
            "image_filename": img_path.name,
            "mask_filename": mask_path.name,
            "tumour_label": row.get("tumour_label", "unknown"),
            "plane_label": row.get("plane_label", "unknown"),
            "class_id": int(row["class_id"]) if "class_id" in row and pd.notna(row["class_id"]) else -1,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = ConvBlock(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3, pos_weight=4.0, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.pos_weight_value = pos_weight

    def forward(self, logits, target):
        pos_weight = torch.tensor([self.pos_weight_value], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce


def postprocess_pred_mask(pred_mask, min_component_area=30, keep_largest=True, fill_holes=True):
    mask = pred_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask.astype(bool)

    comps = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            comps.append((label_id, area))

    if not comps:
        return np.zeros_like(mask).astype(bool)

    clean = np.zeros_like(mask)

    if keep_largest:
        largest = max(comps, key=lambda x: x[1])[0]
        clean[labels == largest] = 1
    else:
        for label_id, _ in comps:
            clean[labels == label_id] = 1

    if fill_holes:
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(clean)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        clean = filled

    return clean.astype(bool)


@torch.no_grad()
def evaluate_model(model, loader, device, epoch, split_name, threshold=0.5, save_per_sample=False, per_sample_dir=None):
    model.eval()
    rows = []

    for batch in tqdm(loader, desc=f"Evaluating {split_name} epoch {epoch}", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].cpu().numpy()

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs >= threshold).astype(np.uint8)

        for i in range(len(preds)):
            pred_mask = preds[i, 0]
            gt_mask = masks[i, 0].astype(bool)

            pred_mask = postprocess_pred_mask(pred_mask)

            metrics = compute_binary_segmentation_metrics(
                pred=pred_mask,
                gt=gt_mask,
                pred_threshold=None,
                gt_threshold=None,
                spacing=None,
            )

            rows.append({
                "epoch": epoch,
                "split": split_name,
                "image_filename": batch["image_filename"][i],
                "mask_filename": batch["mask_filename"][i],
                "tumour_label": batch["tumour_label"][i],
                "plane_label": batch["plane_label"][i],
                "gt_class_id": int(batch["class_id"][i]),
                **metrics,
            })

    sample_df = pd.DataFrame(rows)

    if save_per_sample and per_sample_dir is not None:
        out_dir = per_sample_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(out_dir / f"{split_name}_sample_metrics_epoch_{epoch:03d}.csv", index=False)

    summary = {f"{split_name}_num_images": int(len(sample_df))}

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


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running += loss.item() * images.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def compute_val_loss(model, loader, criterion, device):
    model.eval()
    running = 0.0

    for batch in tqdm(loader, desc="Validation loss", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running += loss.item() * images.size(0)

    return running / len(loader.dataset)


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
    merged = results_df.merge(custom_df, on="epoch", how="left")
    merged.to_csv(out_path, index=False)


def make_eval_subset(csv_path, out_path, limit):
    df = pd.read_csv(csv_path)
    if limit and limit > 0:
        df = df.head(limit)
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)

    parser.add_argument("--project", type=str, default="experiments/brisc_attention_unet")
    parser.add_argument("--name", type=str, default="AttentionUNet-Binary-512")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)

    parser.add_argument("--mask-threshold", type=int, default=1)
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=4.0)

    parser.add_argument("--train-eval-limit", type=int, default=300)
    parser.add_argument("--val-eval-limit", type=int, default=0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-per-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-epoch-weights", action="store_true")

    args = parser.parse_args()
    seed_everything(args.seed)

    run_dir = Path(args.project) / args.name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    results_path = run_dir / "results.csv"
    custom_metrics_path = run_dir / "custom_train_val_metrics_by_epoch.csv"
    merged_path = run_dir / "research_metrics_merged.csv"
    per_sample_dir = run_dir / "per_sample_train_val_metrics"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = BriscBinaryDataset(args.train_csv, args.imgsz, args.mask_threshold, train=True)
    val_dataset = BriscBinaryDataset(args.val_csv, args.imgsz, args.mask_threshold, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    train_eval_csv = make_eval_subset(args.train_csv, run_dir / "train_eval_subset.csv", args.train_eval_limit)
    val_eval_csv = make_eval_subset(args.val_csv, run_dir / "val_eval_subset.csv", args.val_eval_limit)

    train_eval_dataset = BriscBinaryDataset(train_eval_csv, args.imgsz, args.mask_threshold, train=False)
    val_eval_dataset = BriscBinaryDataset(val_eval_csv, args.imgsz, args.mask_threshold, train=False)

    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_eval_loader = DataLoader(val_eval_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = AttentionUNet(base=args.base_channels).to(device)

    criterion = DiceBCELoss(
        dice_weight=0.7,
        bce_weight=0.3,
        pos_weight=args.pos_weight,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_val_dice = -1.0
    best_epoch = -1
    bad_epochs = 0
    results_rows = []
    custom_rows = []

    print("=" * 80)
    print("ATTENTION U-NET BRISC RESEARCH TRAINING")
    print("=" * 80)
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Run dir: {run_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Device: {device}")
    print(f"Base channels: {args.base_channels}")
    print(f"LR: {args.lr}")
    print(f"Pos weight: {args.pos_weight}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = compute_val_loss(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        results_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })
        pd.DataFrame(results_rows).to_csv(results_path, index=False)

        train_summary = evaluate_model(model, train_eval_loader, device, epoch, "train", args.pred_threshold)
        val_summary = evaluate_model(
            model,
            val_eval_loader,
            device,
            epoch,
            "val",
            args.pred_threshold,
            save_per_sample=args.save_per_sample,
            per_sample_dir=per_sample_dir,
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

        save_checkpoint(weights_dir / "last.pt", model, optimizer, epoch, best_val_dice)

        if args.save_epoch_weights:
            save_checkpoint(weights_dir / f"epoch{epoch}.pt", model, optimizer, epoch, best_val_dice)

        val_dice = val_summary["val_dice"]

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(weights_dir / "best.pt", model, optimizer, epoch, best_val_dice)
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