import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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

        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        # IMPORTANT: your masks are 0/1, not 0/255
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
            angle = np.random.uniform(-8, 8)
            scale = np.random.uniform(0.95, 1.05)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(
                mask, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT
            )

        if np.random.rand() < 0.35:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-8, 8)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        if np.random.rand() < 0.25:
            noise = np.random.normal(0, 4, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img, mask


# -----------------------------
# JEPA Encoder - same as pretrain v2
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
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.05,
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
# Attention U-Net Decoder
# -----------------------------
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
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        attn = self.relu(self.W_g(g) + self.W_x(x))
        attn = self.psi(attn)
        return x * attn


class JEPAAttentionUNet(nn.Module):
    def __init__(self, encoder, embed_dim=384):
        super().__init__()
        self.encoder = encoder

        # CNN stem for local skip features
        self.s1 = ConvBlock(1, 32)       # 256x256
        self.p1 = nn.MaxPool2d(2)

        self.s2 = ConvBlock(32, 64)      # 128x128
        self.p2 = nn.MaxPool2d(2)

        self.s3 = ConvBlock(64, 128)     # 64x64
        self.p3 = nn.MaxPool2d(2)

        self.s4 = ConvBlock(128, 256)    # 32x32
        self.p4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(embed_dim, 512)  # 16x16 from JEPA

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = AttentionGate(32, 32, 16)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.s1(x)
        x1 = self.p1(s1)

        s2 = self.s2(x1)
        x2 = self.p2(s2)

        s3 = self.s3(x2)
        x3 = self.p3(s3)

        s4 = self.s4(x3)

        j = self.encoder(x)
        b = self.bottleneck(j)

        d4 = self.up4(b)
        s4 = self.att4(d4, s4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(d3, s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(d2, s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(d1, s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.out(d1)


# -----------------------------
# Losses
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

    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - score.mean()


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    loss = alpha * ((1 - pt) ** gamma) * bce
    return loss.mean()


def total_loss_fn(logits, targets):
    dloss = dice_loss(logits, targets)
    tloss = tversky_loss(logits, targets)
    floss = focal_loss(logits, targets)
    loss = 0.4 * dloss + 0.4 * tloss + 0.2 * floss
    return loss, dloss.item(), tloss.item(), floss.item()


# -----------------------------
# Metrics
# -----------------------------
def hd95_asd(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0, 0.0

    if pred.sum() == 0 or target.sum() == 0:
        return 256.0, 256.0

    kernel = np.ones((3, 3), np.uint8)

    pred_surface = pred - cv2.erode(pred, kernel)
    target_surface = target - cv2.erode(target, kernel)

    pred_dist = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 5)
    target_dist = cv2.distanceTransform(1 - target, cv2.DIST_L2, 5)

    d1 = target_dist[pred_surface > 0]
    d2 = pred_dist[target_surface > 0]

    if len(d1) == 0 or len(d2) == 0:
        return 256.0, 256.0

    d = np.concatenate([d1, d2])
    return float(np.percentile(d, 95)), float(np.mean(d))


def calculate_metrics(logits, targets, threshold=0.3):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    dices, ious, precisions, recalls, hd95s, asds = [], [], [], [], [], []
    fg_ratios = []

    for p, t in zip(probs, targets):
        pred = (p[0] > threshold).astype(np.uint8)
        gt = (t[0] > 0.5).astype(np.uint8)

        tp = np.logical_and(pred == 1, gt == 1).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        h, a = hd95_asd(pred, gt)

        dices.append(dice)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        hd95s.append(h)
        asds.append(a)
        fg_ratios.append(pred.mean())

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "hd95": float(np.mean(hd95s)),
        "asd": float(np.mean(asds)),
        "pred_fg_ratio": float(np.mean(fg_ratios)),
    }


# -----------------------------
# Train / Val
# -----------------------------
def run_epoch(model, loader, optimizer, scaler, device, train=True, threshold=0.3):
    model.train(train)

    total_loss = 0.0
    total_dice_loss = 0.0
    total_tversky_loss = 0.0
    total_focal_loss = 0.0

    metric_store = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "hd95": [],
        "asd": [],
        "pred_fg_ratio": [],
    }

    pbar = tqdm(loader, desc="Train" if train else "Val")

    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss, dloss, tloss, floss = total_loss_fn(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice_loss += dloss * bs
        total_tversky_loss += tloss * bs
        total_focal_loss += floss * bs

        m = calculate_metrics(logits, masks, threshold=threshold)
        for k in metric_store:
            metric_store[k].append(m[k])

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{m['dice']:.4f}",
            fg=f"{m['pred_fg_ratio']:.4f}"
        )

    n = len(loader.dataset)

    out = {
        "loss": total_loss / n,
        "dice_loss": total_dice_loss / n,
        "tversky_loss": total_tversky_loss / n,
        "focal_loss": total_focal_loss / n,
    }

    for k, v in metric_store.items():
        out[k] = float(np.mean(v))

    return out


# -----------------------------
# Plots and overlays
# -----------------------------
def save_plots(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)

    metrics = [
        "loss", "dice_loss", "dice", "iou",
        "precision", "recall", "hd95", "asd", "pred_fg_ratio"
    ]

    for m in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df[f"train_{m}"], label=f"Train {m} {df[f'train_{m}'].iloc[-1]:.3f}")
        plt.plot(df["epoch"], df[f"val_{m}"], label=f"Val {m} {df[f'val_{m}'].iloc[-1]:.3f}")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title("JEPA + attention U-net")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{m}.png", dpi=300)
        plt.close()


def save_overlay_samples(model, loader, device, out_dir, threshold=0.3, max_samples=30):
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
                pred = (probs[i, 0] > threshold).astype(np.uint8)

                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # green = GT, red = prediction, yellow = overlap
                rgb[gt == 1] = [0, 255, 0]
                rgb[pred == 1] = [0, 0, 255]
                rgb[(gt == 1) & (pred == 1)] = [0, 255, 255]

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
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--lr-decoder", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = BRISCSegDataset(args.data, split="train", imgsz=args.imgsz, augment=True)
    val_ds = BRISCSegDataset(args.data, split="val", imgsz=args.imgsz, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    encoder = ViTEncoder(
        img_size=args.imgsz,
        patch_size=args.patch,
        in_chans=1,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
    )

    state = torch.load(args.jepa, map_location="cpu")
    encoder.load_state_dict(state, strict=True)

    model = JEPAAttentionUNet(encoder, embed_dim=args.embed_dim).to(device)

    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_decoder,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice = -1.0
    history = []
    csv_path = out_dir / "research_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        if epoch == args.freeze_epochs + 1:
            print("Unfreezing JEPA encoder...")
            for p in model.encoder.parameters():
                p.requires_grad = True

            encoder_params = list(model.encoder.parameters())
            decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]

            optimizer = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": args.lr_encoder},
                    {"params": decoder_params, "lr": args.lr_decoder},
                ],
                weight_decay=args.weight_decay,
            )

        train_m = run_epoch(
            model, train_loader, optimizer, scaler, device,
            train=True,
            threshold=args.threshold
        )

        val_m = run_epoch(
            model, val_loader, optimizer, scaler, device,
            train=False,
            threshold=args.threshold
        )

        row = {"epoch": epoch}
        for k, v in train_m.items():
            row[f"train_{k}"] = v
        for k, v in val_m.items():
            row[f"val_{k}"] = v

        history.append(row)
        pd.DataFrame(history).to_csv(csv_path, index=False)

        print(
            f"Train Dice: {train_m['dice']:.4f} | "
            f"Val Dice: {val_m['dice']:.4f} | "
            f"Val IoU: {val_m['iou']:.4f}"
        )

        print(
            f"Val Precision: {val_m['precision']:.4f} | "
            f"Val Recall: {val_m['recall']:.4f} | "
            f"Val HD95: {val_m['hd95']:.4f} | "
            f"Val ASD: {val_m['asd']:.4f} | "
            f"Val FG Ratio: {val_m['pred_fg_ratio']:.5f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "args": vars(args),
                "val_dice": val_m["dice"],
            },
            out_dir / "last_jepa_attention_unet.pth",
        )

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "val_dice": val_m["dice"],
                },
                out_dir / "best_jepa_attention_unet.pth",
            )
            print("Best model saved.")

        save_plots(csv_path, out_dir)
        save_overlay_samples(
            model,
            val_loader,
            device,
            out_dir,
            threshold=args.threshold,
            max_samples=30,
        )

    print("\nTraining completed.")
    print(f"Best Val Dice: {best_dice:.6f}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()