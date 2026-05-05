from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from runner_common import (
    canonical_partition_name,
    canonical_trigger_name,
    configure_partition,
    configure_trigger,
    dataset_from_config_name,
    fraction_tag,
    normalize_requested_malicious_fractions,
    normalize_requested_partitions,
    normalize_requested_triggers,
    output_dir_is_new_enough,
)
from table_output_layout import build_direct_output_root

@dataclass(frozen=True)
class DatasetSpec:
    label: str
    slug: str
    config_name: str


DATASETS = [
    DatasetSpec("Cora", "cora", "Cora"),
    DatasetSpec("PubMed", "pubmed", "PubMed"),
    DatasetSpec("CiteSeer", "citeseer", "CiteSeer"),
    DatasetSpec("Coauthor CS", "coauthor_cs", "CoauthorCS"),
    DatasetSpec("Amz-Photo", "amz_photo", "AmazonPhoto"),
    DatasetSpec("Coauthor-Ph", "coauthor_ph", "CoauthorPhysics"),
]
PARTITIONS = ("iid", "louvain", "label_skew", "feature_skew")
TRIGGERS = ("renyi", "ba", "ws", "gta")
MALICIOUS_FRACTIONS = (0.1, 0.3, 0.5)
SEEDS = (42, 43, 44, 45, 46)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def completed_run_exists(
    outputs_dir: Path,
    spec: DatasetSpec,
    partition: str,
    trigger: str,
    seed: int,
    rounds: int,
    malicious_fraction: float,
) -> bool:
    for config_path in sorted(outputs_dir.glob("*/config.json"), reverse=True):
        if not output_dir_is_new_enough(config_path.parent):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(config.get("seed", -1)) != int(seed):
            continue
        if canonical_partition_name(config.get("federated", {}).get("partition", "")) != canonical_partition_name(partition):
            continue
        if canonical_trigger_name(config.get("attack", {}).get("name", "")) != canonical_trigger_name(trigger):
            continue
        if abs(float(config.get("attack", {}).get("malicious_client_fraction", 0.3)) - float(malicious_fraction)) > 1e-12:
            continue
        if dataset_from_config_name(config.get("dataset", {}).get("name", "")) != spec.label:
            continue
        metrics_path = config_path.parent / "metrics.csv"
        if not metrics_path.exists():
            continue
        try:
            metrics = pd.read_csv(metrics_path, usecols=["round"])
        except Exception:
            continue
        if not metrics.empty and int(metrics["round"].max()) >= int(rounds):
            return True
    return False


def build_config(base_cfg: dict, spec: DatasetSpec, partition: str, trigger: str, seed: int, malicious_fraction: float) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["seed"] = int(seed)
    cfg["dataset"]["name"] = spec.config_name
    cfg["dataset"]["root"] = "data"
    partition = configure_partition(
        cfg,
        spec_slug=spec.slug,
        partition=partition,
        seed=int(seed),
        project_root=Path(__file__).resolve().parent,
        metadata_absolute=False,
    )
    trigger = configure_trigger(cfg, trigger)
    cfg["attack"]["malicious_client_fraction"] = float(malicious_fraction)
    cfg["logging"]["restore_best_checkpoint"] = False
    cfg["logging"]["output_root"] = str(
        build_direct_output_root(
            method="main",
            dataset_name=spec.config_name,
            partition=partition,
            trigger=cfg["attack"]["name"],
            malicious_fraction=float(malicious_fraction),
            alpha=cfg["federated"].get("alpha"),
        )
    )
    return cfg


def selected_specs(names: Iterable[str] | None) -> list[DatasetSpec]:
    if not names:
        return list(DATASETS)
    wanted = {name.strip().lower() for name in names}
    picked = []
    for spec in DATASETS:
        aliases = {spec.label.lower(), spec.slug.lower(), spec.config_name.lower()}
        if wanted.intersection(aliases):
            picked.append(spec)
    missing = sorted(wanted - {item for spec in picked for item in (spec.label.lower(), spec.slug.lower(), spec.config_name.lower())})
    if missing:
        raise SystemExit(f"Unknown dataset selector(s): {', '.join(missing)}")
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full VeriDrift main experiment sweep.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conda-env", default="veridrift")
    parser.add_argument("--force", action="store_true", help="Rerun even when a complete matching output exists.")
    parser.add_argument("--datasets", nargs="*", help="Optional subset by label, slug, or config name.")
    parser.add_argument("--partitions", nargs="*", default=["iid", "louvain"])
    parser.add_argument("--triggers", nargs="*", default=["renyi"])
    parser.add_argument("--malicious-fractions", nargs="*", type=float, default=[0.3])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = parser.parse_args()
    args.partitions = normalize_requested_partitions(args.partitions, PARTITIONS)
    args.triggers = normalize_requested_triggers(args.triggers, TRIGGERS)
    args.malicious_fractions = normalize_requested_malicious_fractions(args.malicious_fractions)

    project_root = Path(__file__).resolve().parent
    base_cfg = load_yaml(project_root / "configs" / "cora_node_iid_renyi.yaml")
    config_dir = project_root / "configs" / "generated_main"
    specs = selected_specs(args.datasets)

    total = len(specs) * len(args.partitions) * len(args.triggers) * len(args.malicious_fractions) * len(args.seeds)
    index = 0
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for partition in args.partitions:
        for trigger in args.triggers:
            for malicious_fraction in args.malicious_fractions:
                for spec in specs:
                    for seed in args.seeds:
                        index += 1
                        cfg = build_config(
                            base_cfg,
                            spec=spec,
                            partition=partition,
                            trigger=trigger,
                            seed=int(seed),
                            malicious_fraction=float(malicious_fraction),
                        )
                        rounds = int(cfg["training"]["global_rounds"])
                        outputs_dir = Path(str(cfg["logging"]["output_root"]))
                        frac_tag = fraction_tag(malicious_fraction)
                        config_path = config_dir / f"{spec.slug}_node_{partition}_{frac_tag}_seed{seed}_{trigger}.yaml"
                        save_yaml(cfg, config_path)

                        prefix = (
                            f"[{index:03d}/{total:03d}] {partition} + {trigger} "
                            f"{spec.label} seed={seed} y={malicious_fraction:.1f}"
                        )
                        if not args.force and completed_run_exists(
                            outputs_dir,
                            spec,
                            partition,
                            trigger,
                            int(seed),
                            rounds,
                            float(malicious_fraction),
                        ):
                            print(f"{prefix} complete output exists, skipping.", flush=True)
                            continue

                        print(f"{prefix} starting.", flush=True)
                        if os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env or shutil.which("conda") is None:
                            cmd = [
                                sys.executable,
                                "train_fgl_node_risk.py",
                                "--config",
                                str(config_path),
                                "--device",
                                args.device,
                            ]
                        else:
                            cmd = [
                                shutil.which("conda") or "conda",
                                "run",
                                "-n",
                                args.conda_env,
                                "python",
                                "train_fgl_node_risk.py",
                                "--config",
                                str(config_path),
                                "--device",
                                args.device,
                            ]
                        result = subprocess.run(cmd, cwd=project_root, env=env)
                        if result.returncode != 0:
                            raise SystemExit(result.returncode)
                        print(f"{prefix} finished.", flush=True)


if __name__ == "__main__":
    main()
