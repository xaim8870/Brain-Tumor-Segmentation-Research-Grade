from pathlib import Path
import yaml


def load_yaml_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def get_nested(config: dict, keys: list[str], default=None):
    current = config

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def resolve_project_path(repo_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return repo_root / path