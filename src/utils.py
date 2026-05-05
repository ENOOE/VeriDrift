from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parent


def resolve_project_path(raw_path: str | Path | None, project_root: Path) -> Path | None:
    if raw_path is None:
        return None

    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        return (project_root / path).resolve()
    if path.exists() or (path.is_absolute() and path.parent.exists()):
        return path

    parts = list(path.parts)
    for anchor in ("data", "outputs", "partitions", "analysis_outputs"):
        if anchor in parts:
            anchor_index = parts.index(anchor)
            suffix = Path(*parts[anchor_index + 1 :]) if anchor_index + 1 < len(parts) else Path()
            return (project_root / anchor / suffix).resolve()
    return (project_root / path.name).resolve()


def normalize_runtime_paths(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    normalized = dict(cfg)

    dataset_cfg = dict(normalized.get("dataset", {}))
    dataset_cfg["root"] = str(resolve_project_path(dataset_cfg.get("root", "data"), project_root))
    normalized["dataset"] = dataset_cfg

    logging_cfg = dict(normalized.get("logging", {}))
    logging_cfg["output_root"] = str(resolve_project_path(logging_cfg.get("output_root", "outputs"), project_root))
    normalized["logging"] = logging_cfg

    federated_cfg = dict(normalized.get("federated", {}))
    partition_path = federated_cfg.get("partition_metadata_path")
    if partition_path:
        federated_cfg["partition_metadata_path"] = str(resolve_project_path(partition_path, project_root))
    normalized["federated"] = federated_cfg
    return normalized


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
