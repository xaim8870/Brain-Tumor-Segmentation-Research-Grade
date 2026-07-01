import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


# -----------------------------
# Dataset
# -----------------------------
class BRISCSegDataset(Dataset):
    def __init__(self, data_dir, split="train", imgsz=256, augment=False):
        self.img_dir = Path(data_dir) / "images" / split
        self.mask_dir = Path(data_dir) / "masks" / split
        self.imgsz = imgsz
        self.augment = augment

        self.images = sorted([
            p for p in self.img_dir.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        ])

        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(img_path)
        if mask is None:
            raise FileNotFoundError(mask_path)

        img = cv2.resize(img, (self.imgsz, self.imgsz))
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)

        if self.augment:
            img, mask = self.apply_aug(img, mask)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        img = torch.tensor(img).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask, img_path.name

    def apply_aug(self, img, mask):
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        if np.random.rand() < 0.4:
            angle = np.random.uniform(-10, 10)
            scale = np.random.uniform(0.95, 1.05)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-10, 10)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img, mask


# -----------------------------
# JEPA Encoder same as pretrain
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)

        b, n, c = x.shape
        h = w = self.grid_size
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


# -----------------------------
# U-Net Decoder
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class JEPAUNetSeg(nn.Module):
    def __init__(self, encoder, embed_dim=384):
        super().__init__()
        self.encoder = encoder

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBlock(embed_dim, 256))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBlock(256, 128))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBlock(128, 64))
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvBlock(64, 32))

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out(x)


# -----------------------------
# Loss and Metrics
# -----------------------------
def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1 - dice.mean()


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky.mean()


def bce_dice_loss(logits, targets):
    pos_weight = torch.tensor([20.0], device=targets.device)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight
    )

    dloss = dice_loss(logits, targets)
    tloss = tversky_loss(logits, targets)

    loss = 0.2 * bce + 0.4 * dloss + 0.4 * tloss
    return loss, bce.item(), dloss.item()

def metrics_from_logits(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.3).float()

    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    dices, ious, precs, recalls = [], [], [], []

    for p, t in zip(preds, targets):
        p = p[0].astype(np.uint8)
        t = t[0].astype(np.uint8)

        tp = np.logical_and(p == 1, t == 1).sum()
        fp = np.logical_and(p == 1, t == 0).sum()
        fn = np.logical_and(p == 0, t == 1).sum()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        dices.append(dice)
        ious.append(iou)
        precs.append(precision)
        recalls.append(recall)

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recalls)),
    }


def hd95_asd(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    if pred.sum() == 0 or target.sum() == 0:
        return 0.0, 0.0

    pred_dist = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 5)
    target_dist = cv2.distanceTransform(1 - target, cv2.DIST_L2, 5)

    pred_surface = pred - cv2.erode(pred, np.ones((3, 3), np.uint8))
    target_surface = target - cv2.erode(target, np.ones((3, 3), np.uint8))

    d1 = target_dist[pred_surface > 0]
    d2 = pred_dist[target_surface > 0]

    if len(d1) == 0 or len(d2) == 0:
        return 0.0, 0.0

    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95)), float(np.mean(all_d))


# -----------------------------
# Train / Val
# -----------------------------
def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)

    total_loss = 0
    total_bce = 0
    total_dice_loss = 0

    all_dice, all_iou, all_precision, all_recall = [], [], [], []
    all_hd95, all_asd = [], []

    pbar = tqdm(loader, desc="Train" if train else "Val")

    for imgs, masks, names in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss, bce, dloss = bce_dice_loss(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_metrics = metrics_from_logits(logits, masks)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        gt = masks.detach().cpu().numpy()

        for p, t in zip(probs, gt):
            p = (p[0] > 0.3).astype(np.uint8)
            t = (t[0] > 0.5).astype(np.uint8)
            h, a = hd95_asd(p, t)
            all_hd95.append(h)
            all_asd.append(a)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_bce += bce * bs
        total_dice_loss += dloss * bs

        all_dice.append(batch_metrics["dice"])
        all_iou.append(batch_metrics["iou"])
        all_precision.append(batch_metrics["precision"])
        all_recall.append(batch_metrics["recall"])

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_metrics['dice']:.4f}")

    n = len(loader.dataset)

    return {
        "loss": total_loss / n,
        "bce_loss": total_bce / n,
        "dice_loss": total_dice_loss / n,
        "dice": float(np.mean(all_dice)),
        "iou": float(np.mean(all_iou)),
        "precision": float(np.mean(all_precision)),
        "recall": float(np.mean(all_recall)),
        "hd95": float(np.mean(all_hd95)),
        "asd": float(np.mean(all_asd)),
    }


def save_plots(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics = ["loss", "dice_loss", "dice", "iou", "precision", "recall", "hd95", "asd"]

    for m in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df[f"train_{m}"], label=f"Train {m} {df[f'train_{m}'].iloc[-1]:.3f}")
        plt.plot(df["epoch"], df[f"val_{m}"], label=f"Val {m} {df[f'val_{m}'].iloc[-1]:.3f}")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title("JEPA Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{m}.png", dpi=300)
        plt.close()


def save_overlay_samples(model, loader, device, out_dir, max_samples=20):
    overlay_dir = Path(out_dir) / "overlay_predictions"
    overlay_dir.mkdir(exist_ok=True)

    model.eval()
    saved = 0

    with torch.no_grad():
        for imgs, masks, names in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()

            imgs_np = imgs.cpu().numpy()
            masks_np = masks.numpy()

            for i in range(len(names)):
                img = imgs_np[i, 0]
                img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

                gt = (masks_np[i, 0] > 0.5).astype(np.uint8)
                pred = (probs[i, 0] > 0.3).astype(np.uint8)

                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                rgb[gt == 1] = [0, 255, 0]
                rgb[pred == 1] = [0, 0, 255]

                cv2.imwrite(str(overlay_dir / names[i]), rgb)

                saved += 1
                if saved >= max_samples:
                    return


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--jepa", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--lr-decoder", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--freeze-epochs", type=int, default=10)

    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--patch", type=int, default=16)

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = BRISCSegDataset(args.data, split="train", imgsz=args.imgsz, augment=True)
    val_ds = BRISCSegDataset(args.data, split="val", imgsz=args.imgsz, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    encoder = ViTEncoder(
        img_size=args.imgsz,
        patch_size=args.patch,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
    )

    ckpt = torch.load(args.jepa, map_location="cpu")
    encoder.load_state_dict(ckpt, strict=True)

    model = JEPAUNetSeg(encoder, embed_dim=args.embed_dim).to(device)

    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_decoder,
        weight_decay=1e-4,
    )

    best_dice = -1
    history = []
    csv_path = out_dir / "research_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        if epoch == args.freeze_epochs + 1:
            print("Unfreezing JEPA encoder...")
            for p in model.encoder.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": args.lr_encoder},
                {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": args.lr_decoder},
            ], weight_decay=1e-4)

        train_m = run_epoch(model, train_loader, optimizer, device, train=True)
        val_m = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {"epoch": epoch}
        for k, v in train_m.items():
            row[f"train_{k}"] = v
        for k, v in val_m.items():
            row[f"val_{k}"] = v

        history.append(row)
        pd.DataFrame(history).to_csv(csv_path, index=False)

        print(f"Train Dice: {train_m['dice']:.4f} | Val Dice: {val_m['dice']:.4f}")
        print(f"Val IoU: {val_m['iou']:.4f} | Val HD95: {val_m['hd95']:.4f} | Val ASD: {val_m['asd']:.4f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "val_dice": val_m["dice"],
        }, out_dir / "last_jepa_segmentation.pth")

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "args": vars(args),
                "val_dice": val_m["dice"],
            }, out_dir / "best_jepa_segmentation.pth")
            print("Best model saved.")

        save_plots(csv_path, out_dir)
        save_overlay_samples(model, val_loader, device, out_dir, max_samples=20)

    print("\nTraining finished.")
    print("Best Val Dice:", best_dice)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()