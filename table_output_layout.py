from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ORGANIZED_RESULTS_ROOT = Path("outputs/organized_table_results")


@dataclass(frozen=True)
class TableSpec:
    slug: str
    partition: str
    malicious_fraction: float
    trigger: str
    alpha: float | None = None


TABLE_SPECS = [
    TableSpec("T01_iid_y03_renyi", "iid", 0.3, "renyi"),
    TableSpec("T02_louvain_y03_renyi", "louvain", 0.3, "renyi"),
    TableSpec("T03_iid_y05_renyi", "iid", 0.5, "renyi"),
    TableSpec("T04_louvain_y05_renyi", "louvain", 0.5, "renyi"),
    TableSpec("T05_iid_y01_renyi", "iid", 0.1, "renyi"),
    TableSpec("T06_louvain_y01_renyi", "louvain", 0.1, "renyi"),
    TableSpec("T07_label_skew_a05_y03_renyi", "label_skew", 0.3, "renyi", alpha=0.5),
    TableSpec("T08_feature_skew_a05_y03_renyi", "feature_skew", 0.3, "renyi", alpha=0.5),
    TableSpec("T09_louvain_y03_gta", "louvain", 0.3, "gta"),
    TableSpec("T10_louvain_y03_ba", "louvain", 0.3, "ba"),
    TableSpec("T11_louvain_y03_ws", "louvain", 0.3, "ws"),
]


def canonical_dataset_name(name: str | None) -> str:
    key = str(name or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "cora": "Cora",
        "pubmed": "PubMed",
        "citeseer": "CiteSeer",
        "coauthorcs": "CoauthorCS",
        "coauthorphysics": "CoauthorPhysics",
        "coauthorph": "CoauthorPhysics",
        "amazonphoto": "AmazonPhoto",
        "amzphoto": "AmazonPhoto",
    }
    return aliases.get(key, str(name or "unknown"))


def canonical_partition_name(name: str | None) -> str:
    key = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "iid": "iid",
        "louvain": "louvain",
        "dirichlet": "label_skew",
        "non_iid_label_skew": "label_skew",
        "label_skew": "label_skew",
        "non_iid_feature_skew": "feature_skew",
        "feature_skew": "feature_skew",
    }
    return aliases.get(key, key)


def canonical_trigger_name(name: str | None) -> str:
    key = str(name or "").strip().lower()
    aliases = {
        "multi_client_renyi_trigger": "renyi",
        "renyi": "renyi",
        "multi_client_gta_trigger": "gta",
        "gta": "gta",
        "multi_client_ba_trigger": "ba",
        "ba": "ba",
        "multi_client_ws_trigger": "ws",
        "ws": "ws",
    }
    return aliases.get(key, key)


def canonical_method_name(name: str | None) -> str:
    key = str(name or "main").strip().lower()
    aliases = {
        "": "main",
        "main": "main",
        "fltrust": "fltrust",
        "rfa": "rfa",
        "rlr": "rlr",
        "dnc": "dnc",
        "flame": "flame",
        "fedcpa": "fedcpa",
        "fedavg": "fedavg",
        "fedtge": "fedtge",
    }
    return aliases.get(key, key)


def resolve_table_slug(
    partition: str,
    trigger: str,
    malicious_fraction: float,
    alpha: float | None = None,
) -> str:
    for spec in TABLE_SPECS:
        if spec.partition != partition:
            continue
        if spec.trigger != trigger:
            continue
        if abs(float(spec.malicious_fraction) - float(malicious_fraction)) >= 1e-12:
            continue
        if spec.alpha is None:
            return spec.slug
        if alpha is not None and abs(float(spec.alpha) - float(alpha)) < 1e-12:
            return spec.slug
    alpha_tag = "na" if alpha is None else f"a{str(alpha).replace('.', '')}"
    frac_tag = str(malicious_fraction).replace(".", "")
    return f"_unmapped/{partition}_y{frac_tag}_{trigger}_{alpha_tag}"


def build_direct_output_root(
    *,
    method: str,
    dataset_name: str,
    partition: str,
    trigger: str,
    malicious_fraction: float,
    alpha: float | None = None,
) -> Path:
    table_slug = resolve_table_slug(
        partition=canonical_partition_name(partition),
        trigger=canonical_trigger_name(trigger),
        malicious_fraction=float(malicious_fraction),
        alpha=alpha,
    )
    return (
        ORGANIZED_RESULTS_ROOT
        / table_slug
        / "all_runs"
        / canonical_method_name(method)
        / canonical_dataset_name(dataset_name)
    )
