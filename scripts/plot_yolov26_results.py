import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path(r"D:\Brain Tumor Segmentation\results\research_metrics_merged.csv")
out_dir = csv_path.parent / "yolov26_research_metrics_plots"
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

epoch_col = "epoch_ultralytics" if "epoch_ultralytics" in df.columns else "epoch"

percent_cols = [
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)",
    "train_dice", "val_dice", "train_iou", "val_iou",
    "train_precision", "val_precision", "train_recall", "val_recall",
    "train_sensitivity", "val_sensitivity",
    "train_specificity", "val_specificity",
    "train_volume_similarity", "val_volume_similarity"
]

plot_groups = {
    "Box Metrics": ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
    "Mask Metrics": ["metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"],
    "Training Losses": ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss", "train/sem_loss"],
    "Validation Losses": ["val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss", "val/sem_loss"],
    "Medical Dice": ["train_dice", "val_dice"],
    "Medical IoU": ["train_iou", "val_iou"],
    "Medical Precision Recall": ["train_precision", "val_precision", "train_recall", "val_recall"],
    "Medical Sensitivity Specificity": ["train_sensitivity", "val_sensitivity", "train_specificity", "val_specificity"],
    "Medical Distance Metrics": ["train_hd95", "val_hd95", "train_asd", "val_asd"],
    "Class Dice": ["val_glioma_dice", "val_meningioma_dice", "val_pituitary_dice"],
    "Class IoU": ["val_glioma_iou", "val_meningioma_iou", "val_pituitary_iou"],
    "Class HD95": ["val_glioma_hd95", "val_meningioma_hd95", "val_pituitary_hd95"],
    "Class ASD": ["val_glioma_asd", "val_meningioma_asd", "val_pituitary_asd"],
}

for title, cols in plot_groups.items():
    available = [c for c in cols if c in df.columns]
    if not available:
        continue

    plt.figure(figsize=(10, 6))

    for col in available:
        y = pd.to_numeric(df[col], errors="coerce")

        if col in percent_cols:
            y = y * 100

        y = y.round(3)
        final_value = y.dropna().iloc[-1]

        plt.plot(df[epoch_col], y, linewidth=2, label=f"{col}: {final_value:.3f}")

    plt.title(f"Yolov26 {title}", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_name = title.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(out_dir / f"yolov26_{save_name}.png", dpi=300)
    plt.close()

print(f"All plots saved at: {out_dir}")