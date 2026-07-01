import os
import math
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
class BRISCImageDataset(Dataset):
    def __init__(self, image_dir, imgsz=256, train=True):
        self.image_dir = Path(image_dir)
        self.paths = sorted([
            p for p in self.image_dir.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        ])

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        if train:
            self.tfms = transforms.Compose([
                transforms.Resize((imgsz, imgsz)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.tfms(img)
        return img


# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)              # B, C, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x


# -----------------------------
# Transformer Encoder
# -----------------------------
class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x


# -----------------------------
# JEPA Model
# -----------------------------
class JEPAModel(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6,
        predictor_depth=2,
    ):
        super().__init__()

        self.context_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        self.target_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        predictor_layers = []
        for _ in range(predictor_depth):
            predictor_layers.append(nn.Linear(embed_dim, embed_dim))
            predictor_layers.append(nn.GELU())
            predictor_layers.append(nn.LayerNorm(embed_dim))

        predictor_layers.append(nn.Linear(embed_dim, embed_dim))
        self.predictor = nn.Sequential(*predictor_layers)

        self._init_target_encoder()

    def _init_target_encoder(self):
        for p_context, p_target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_target.data.copy_(p_context.data)
            p_target.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        for p_context, p_target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_target.data.mul_(momentum).add_(
                p_context.data,
                alpha=1.0 - momentum
            )

    def forward(self, x, mask_ratio=0.4):
        context_feat = self.context_encoder(x)

        with torch.no_grad():
            target_feat = self.target_encoder(x)

        B, N, C = context_feat.shape
        num_mask = int(N * mask_ratio)

        losses = []

        for b in range(B):
            perm = torch.randperm(N, device=x.device)
            mask_idx = perm[:num_mask]

            visible_feat = context_feat[b:b + 1]
            predicted_feat = self.predictor(visible_feat)

            pred_masked = predicted_feat[:, mask_idx, :]
            target_masked = target_feat[b:b + 1, mask_idx, :]

            loss = F.smooth_l1_loss(pred_masked, target_masked)
            losses.append(loss)

        loss = torch.stack(losses).mean()
        return loss


# -----------------------------
# Train / Validate
# -----------------------------
def run_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    train=True,
    mask_ratio=0.4,
    ema_momentum=0.996,
    grad_accum=1,
):
    model.train(train)

    total_loss = 0.0

    if train:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train" if train else "Val")

    for step, imgs in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = model(imgs, mask_ratio=mask_ratio)

            if train:
                loss_for_backward = loss / grad_accum
                scaler.scale(loss_for_backward).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    model.update_target_encoder(momentum=ema_momentum)

        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / len(loader.dataset)


def save_loss_plot(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label=f"Train Loss {df['train_loss'].iloc[-1]:.3f}")
    plt.plot(df["epoch"], df["val_loss"], label=f"Val Loss {df['val_loss'].iloc[-1]:.3f}")
    plt.xlabel("Epoch")
    plt.ylabel("JEPA Loss")
    plt.title("JEPA Pretraining")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "jepa_pretrain_loss.png", dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--patch", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)

    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--ema", type=float, default=0.996)
    parser.add_argument("--grad-accum", type=int, default=4)

    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir = data_dir / "images" / "train"
    val_img_dir = data_dir / "images" / "val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Train images: {train_img_dir}")
    print(f"Val images: {val_img_dir}")

    train_ds = BRISCImageDataset(train_img_dir, imgsz=args.imgsz, train=True)
    val_ds = BRISCImageDataset(val_img_dir, imgsz=args.imgsz, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = JEPAModel(
        img_size=args.imgsz,
        patch_size=args.patch,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.context_encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    history = []

    csv_path = out_dir / "jepa_pretrain_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            mask_ratio=args.mask_ratio,
            ema_momentum=args.ema,
            grad_accum=args.grad_accum,
        )

        val_loss = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            scaler=scaler,
            train=False,
            mask_ratio=args.mask_ratio,
            ema_momentum=args.ema,
            grad_accum=args.grad_accum,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "imgsz": args.imgsz,
            "mask_ratio": args.mask_ratio,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(csv_path, index=False)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        torch.save(
            {
                "epoch": epoch,
                "context_encoder": model.context_encoder.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
            },
            out_dir / "last_jepa_pretrain.pth"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "context_encoder": model.context_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "val_loss": val_loss,
                },
                out_dir / "best_jepa_pretrain.pth"
            )

            torch.save(
                model.context_encoder.state_dict(),
                out_dir / "best_jepa_encoder_only.pth"
            )

            print("Best JEPA encoder saved.")

        save_loss_plot(csv_path, out_dir)

    print("\nTraining completed.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()