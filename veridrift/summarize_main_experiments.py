from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    aliases: tuple[str, ...]


DATASETS = [
    DatasetSpec("Cora", ("cora",)),
    DatasetSpec("PubMed", ("pubmed",)),
    DatasetSpec("CiteSeer", ("citeseer",)),
    DatasetSpec("Coauthor CS", ("coauthorcs", "coauthor cs", "cs")),
    DatasetSpec("Coauthor-Ph", ("coauthorphysics", "coauthorph", "physics")),
    DatasetSpec("Amz-Photo", ("amazonphoto", "amzphoto", "photo")),
]
PARTITIONS = ("iid", "louvain")
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)


def name_key(name: object) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def display_name(name: object) -> str:
    key = name_key(name)
    for spec in DATASETS:
        if key in {name_key(alias) for alias in spec.aliases}:
            return spec.label
    return str(name)


def latest_complete_run(outputs_dir: Path, dataset: str, partition: str, seed: int, rounds: int = 100) -> Path | None:
    matches: list[Path] = []
    for config_path in outputs_dir.glob("*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(config.get("seed", -1)) != int(seed):
            continue
        if str(config.get("federated", {}).get("partition", "")).lower() != partition:
            continue
        if display_name(config.get("dataset", {}).get("name", "")) != dataset:
            continue
        metrics_path = config_path.parent / "metrics.csv"
        if not metrics_path.exists():
            continue
        try:
            metrics = pd.read_csv(metrics_path, usecols=["round"])
        except Exception:
            continue
        if not metrics.empty and int(metrics["round"].max()) >= rounds:
            matches.append(config_path.parent)
    return sorted(matches, key=lambda path: path.name)[-1] if matches else None


def summarize_run(run_dir: Path, dataset: str, partition: str, seed: int) -> dict:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    metrics["test_accuracy"] = pd.to_numeric(metrics["test_accuracy"], errors="coerce")
    metrics["asr"] = pd.to_numeric(metrics["asr"], errors="coerce")
    last5 = metrics.sort_values("round").tail(5).copy()
    last5["selection_score"] = last5["test_accuracy"] + (1.0 - last5["asr"])
    valid_asr = last5.dropna(subset=["asr"])
    valid_score = last5.dropna(subset=["selection_score"])
    if valid_score.empty:
        selected_row = last5.sort_values(["test_accuracy"], ascending=[False]).iloc[0]
    else:
        selected_row = valid_score.sort_values(["selection_score", "test_accuracy", "asr"], ascending=[False, False, True]).iloc[0]
    best_acc_row = last5.sort_values(["test_accuracy", "asr"], ascending=[False, True]).iloc[0]
    if valid_asr.empty:
        lowest_asr_row = best_acc_row
    else:
        lowest_asr_row = valid_asr.sort_values(["asr", "test_accuracy"], ascending=[True, False]).iloc[0]
    return {
        "partition": partition,
        "dataset": dataset,
        "seed": int(seed),
        "run": run_dir.name,
        "last5_mean_acc": float(last5["test_accuracy"].mean()),
        "last5_mean_asr": float(valid_asr["asr"].mean()) if not valid_asr.empty else float("nan"),
        "last5_mean_score": float(valid_score["selection_score"].mean()) if not valid_score.empty else float("nan"),
        "last5_selected_round": int(selected_row["round"]),
        "last5_selected_score": float(selected_row["selection_score"]) if pd.notna(selected_row["selection_score"]) else float("nan"),
        "last5_selected_acc": float(selected_row["test_accuracy"]),
        "last5_selected_asr": float(selected_row["asr"]) if pd.notna(selected_row["asr"]) else float("nan"),
        "last5_best_acc_round": int(best_acc_row["round"]),
        "last5_best_acc": float(best_acc_row["test_accuracy"]),
        "last5_best_acc_asr": float(best_acc_row["asr"]) if pd.notna(best_acc_row["asr"]) else float("nan"),
        "last5_lowest_asr_round": int(lowest_asr_row["round"]),
        "last5_lowest_asr_acc": float(lowest_asr_row["test_accuracy"]),
        "last5_lowest_asr": float(lowest_asr_row["asr"]) if pd.notna(lowest_asr_row["asr"]) else float("nan"),
        "round100_acc": float(metrics.sort_values("round").iloc[-1]["test_accuracy"]),
        "round100_asr": float(metrics.sort_values("round").iloc[-1]["asr"]),
    }


def fmt_cell(row: pd.Series) -> str:
    return f"score {row['last5_selected_score']:.4f}; acc {row['last5_selected_acc']:.4f}; asr {row['last5_selected_asr']:.4f}"


def write_markdown_table(frame: pd.DataFrame, partition: str, output_path: Path) -> None:
    rows = []
    for seed in SEEDS:
        line = {"seed": seed}
        for spec in DATASETS:
            match = frame[(frame["partition"] == partition) & (frame["dataset"] == spec.label) & (frame["seed"] == seed)]
            line[spec.label] = "" if match.empty else fmt_cell(match.iloc[0])
        rows.append(line)
    table = pd.DataFrame(rows)
    headers = list(table.columns)
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        markdown.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    output_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize main experiment outputs.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for partition in PARTITIONS:
        for spec in DATASETS:
            for seed in SEEDS:
                run_dir = latest_complete_run(args.outputs_dir, spec.label, partition, seed)
                if run_dir is None:
                    missing.append({"partition": partition, "dataset": spec.label, "seed": seed})
                    continue
                rows.append(summarize_run(run_dir, spec.label, partition, seed))

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "main_experiment_summary_long.csv", index=False)
    pd.DataFrame(missing).to_csv(args.output_dir / "main_experiment_missing.csv", index=False)
    for partition in PARTITIONS:
        write_markdown_table(frame, partition, args.output_dir / f"main_experiment_{partition}_table.md")
    print(args.output_dir / "main_experiment_summary_long.csv")


if __name__ == "__main__":
    main()
