from pathlib import Path
import argparse
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(REPO_ROOT))

from src.utils.config import load_yaml_config, get_nested, resolve_project_path


def run_command(command, dry_run: bool = False):
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(str(x) for x in command))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(command, check=True)


def resolve_existing_or_raw(value):
    """
    For model names like yolo26n-seg.pt:
    - If file exists in repo, use full path.
    - Otherwise return the raw string so Ultralytics can handle it.
    """

    if value is None:
        return None

    path = Path(value)

    if path.is_absolute() and path.exists():
        return str(path)

    repo_path = REPO_ROOT / path

    if repo_path.exists():
        return str(repo_path)

    return str(value)


def build_prepare_command(config: dict):
    paths = config["paths"]
    prepare = config["prepare"]
    project = config.get("project", {})

    brisc_root = resolve_project_path(REPO_ROOT, paths["brisc_root"])
    yolo_dataset_root = resolve_project_path(REPO_ROOT, paths["yolo_dataset_root"])
    splits_dir = resolve_project_path(REPO_ROOT, paths["splits_dir"])

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_brisc_yolo_seg.py"),
        "--root",
        str(brisc_root),
        "--out",
        str(yolo_dataset_root),
        "--splits-out",
        str(splits_dir),
        "--val-ratio",
        str(prepare.get("val_ratio", 0.2)),
        "--seed",
        str(project.get("seed", 42)),
        "--threshold",
        str(prepare.get("threshold", 128)),
        "--min-area",
        str(prepare.get("min_area", 20.0)),
        "--epsilon-ratio",
        str(prepare.get("epsilon_ratio", 0.002)),
    ]

    return command


def build_train_command(config: dict):
    paths = config["paths"]
    train = config["train"]
    eval_cfg = config["eval"]

    yolo_yaml = resolve_project_path(REPO_ROOT, paths["yolo_yaml"])
    val_csv = resolve_project_path(REPO_ROOT, paths["val_csv"])

    experiment_project = resolve_project_path(REPO_ROOT, paths["experiment_project"])

    model_weights = resolve_existing_or_raw(paths["model_weights"])

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_yolo26_brisc_research.py"),
        "--model",
        str(model_weights),
        "--data",
        str(yolo_yaml),
        "--val-csv",
        str(val_csv),
        "--project",
        str(experiment_project),
        "--name",
        str(paths["experiment_name"]),
        "--epochs",
        str(train.get("epochs", 100)),
        "--imgsz",
        str(train.get("imgsz", 256)),
        "--batch",
        str(train.get("batch", 4)),
        "--device",
        str(train.get("device", "0")),
        "--workers",
        str(train.get("workers", 4)),
        "--optimizer",
        str(train.get("optimizer", "auto")),
        "--conf",
        str(eval_cfg.get("conf", 0.25)),
        "--iou",
        str(eval_cfg.get("iou", 0.7)),
        "--eval-batch",
        str(eval_cfg.get("eval_batch", 4)),
        "--mask-threshold",
        str(eval_cfg.get("mask_threshold", 128)),
        "--eval-img-limit",
        str(eval_cfg.get("eval_img_limit", 0)),
    ]

    if eval_cfg.get("save_per_sample", False):
        command.append("--save-per-sample")

    if eval_cfg.get("delete_epoch_weights_after_eval", False):
        command.append("--delete-epoch-weights-after-eval")

    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run BRISC YOLOv26 preparation/training pipeline from config file."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/brisc/yolo26_brisc.yaml",
        help="Path to config YAML file.",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="train",
        choices=["prepare", "train", "all"],
        help="Which stage to run.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = parser.parse_args()

    config_path = resolve_project_path(REPO_ROOT, args.config)
    config = load_yaml_config(config_path)

    print("=" * 80)
    print("BRISC YOLOv26 CONFIG PIPELINE")
    print("=" * 80)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Config file: {config_path}")
    print(f"Stage: {args.stage}")
    print("=" * 80)

    if args.stage in ["prepare", "all"]:
        prepare_command = build_prepare_command(config)
        run_command(prepare_command, dry_run=args.dry_run)

    if args.stage in ["train", "all"]:
        train_command = build_train_command(config)
        run_command(train_command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()