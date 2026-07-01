import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
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

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x


class Predictor(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class JEPAV2(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        self.context_encoder = ViTEncoder(img_size, patch_size, 1, embed_dim, depth, num_heads)
        self.target_encoder = ViTEncoder(img_size, patch_size, 1, embed_dim, depth, num_heads)
        self.predictor = Predictor(embed_dim)

        self.num_patches = self.context_encoder.num_patches
        self.embed_dim = embed_dim

        self.init_target()

    def init_target(self):
        for pc, pt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            pt.data.copy_(pc.data)
            pt.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum):
        for pc, pt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            pt.data.mul_(momentum).add_(pc.data, alpha=1.0 - momentum)

    def make_context_mask(self, x_tokens, target_idx):
        x = x_tokens.clone()
        x[:, target_idx, :] = 0.0
        return x

    def forward(self, imgs, mask_ratio=0.35):
        context_tokens = self.context_encoder(imgs)

        with torch.no_grad():
            target_tokens = self.target_encoder(imgs)

        B, N, C = context_tokens.shape
        num_target = max(1, int(N * mask_ratio))

        losses = []

        for b in range(B):
            target_idx = torch.randperm(N, device=imgs.device)[:num_target]

            masked_context = context_tokens[b:b + 1].clone()
            masked_context[:, target_idx, :] = 0.0

            pred_tokens = self.predictor(masked_context)

            pred = pred_tokens[:, target_idx, :]
            target = target_tokens[b:b + 1, target_idx, :].detach()

            loss = F.smooth_l1_loss(pred, target)
            losses.append(loss)

        return torch.stack(losses).mean()


def cosine_lr(base_lr, min_lr, epoch, total_epochs, warmup_epochs):
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def cosine_momentum(base_m, final_m, epoch, total_epochs):
    progress = epoch / total_epochs
    return final_m - 0.5 * (final_m - base_m) * (1 + math.cos(math.pi * progress))


def run_epoch(model, loader, optimizer, scaler, device, train=True, mask_ratio=0.35,
              grad_accum=8, ema_momentum=0.996, clip_grad=1.0):
    model.train(train)

    total_loss = 0.0

    if train:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train" if train else "Val")

    for step, imgs in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = model(imgs, mask_ratio=mask_ratio)

            if train:
                scaler.scale(loss / grad_accum).backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.context_encoder.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), clip_grad)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    model.update_target(ema_momentum)

        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / len(loader.dataset)


def save_plot(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label=f"Train Loss {df['train_loss'].iloc[-1]:.3f}")
    plt.plot(df["epoch"], df["val_loss"], label=f"Val Loss {df['val_loss'].iloc[-1]:.3f}")
    plt.xlabel("Epoch")
    plt.ylabel("JEPA Loss")
    plt.title("JEPA Pretraining V2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "jepa_pretrain_loss.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--patch", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)

    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--grad-accum", type=int, default=8)

    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=0.999)

    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir = data_dir / "images" / "train"
    val_img_dir = data_dir / "images" / "val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Train images:", train_img_dir)
    print("Val images:", val_img_dir)

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

    model = JEPAV2(
        img_size=args.imgsz,
        patch_size=args.patch,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    history = []
    csv_path = out_dir / "jepa_pretrain_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(args.lr, args.min_lr, epoch, args.epochs, args.warmup_epochs)
        ema_momentum = cosine_momentum(args.ema_start, args.ema_end, epoch, args.epochs)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        print(f"\nEpoch [{epoch}/{args.epochs}] | LR: {lr:.8f} | EMA: {ema_momentum:.6f}")

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train=True,
            mask_ratio=args.mask_ratio,
            grad_accum=args.grad_accum,
            ema_momentum=ema_momentum,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train=False,
            mask_ratio=args.mask_ratio,
            grad_accum=args.grad_accum,
            ema_momentum=ema_momentum,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "ema_momentum": ema_momentum,
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
            out_dir / "last_jepa_pretrain.pth",
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
                out_dir / "best_jepa_pretrain.pth",
            )

            torch.save(
                model.context_encoder.state_dict(),
                out_dir / "best_jepa_encoder_only.pth",
            )

            print("Best JEPA encoder saved.")

        save_plot(csv_path, out_dir)

    print("\nJEPA pretraining completed.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()