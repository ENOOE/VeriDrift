from __future__ import annotations

from pathlib import Path
from typing import Iterable


MIN_VALID_OUTPUT_DATE = "20260423"
SUPPORTED_MALICIOUS_FRACTIONS = (0.1, 0.3, 0.5)


def dataset_from_config_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "cora": "Cora",
        "pubmed": "PubMed",
        "citeseer": "CiteSeer",
        "coauthorcs": "Coauthor CS",
        "coauthorphysics": "Coauthor-Ph",
        "coauthorph": "Coauthor-Ph",
        "amazonphoto": "Amz-Photo",
        "amzphoto": "Amz-Photo",
    }
    return aliases.get(key, str(name))


def canonical_partition_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "dirichlet": "label_skew",
        "non_iid_label_skew": "label_skew",
        "label_skew": "label_skew",
        "non_iid_feature_skew": "feature_skew",
        "feature_skew": "feature_skew",
        "iid": "iid",
        "louvain": "louvain",
    }
    return aliases.get(key, key)


def canonical_trigger_name(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "multi_client_renyi_trigger": "renyi",
        "renyi": "renyi",
        "multi_client_ba_trigger": "ba",
        "ba": "ba",
        "multi_client_ws_trigger": "ws",
        "ws": "ws",
        "multi_client_gta_trigger": "gta",
        "gta": "gta",
    }
    return aliases.get(key, key)


def normalize_requested_partitions(names: Iterable[str], supported: Iterable[str]) -> list[str]:
    supported_set = set(supported)
    normalized = [canonical_partition_name(name) for name in names]
    invalid = sorted({name for name in normalized if name not in supported_set})
    if invalid:
        raise SystemExit(f"Unsupported partition(s): {', '.join(invalid)}")
    return normalized


def normalize_requested_triggers(names: Iterable[str], supported: Iterable[str]) -> list[str]:
    supported_set = set(supported)
    normalized = [canonical_trigger_name(name) for name in names]
    invalid = sorted({name for name in normalized if name not in supported_set})
    if invalid:
        raise SystemExit(f"Unsupported trigger(s): {', '.join(invalid)}")
    return normalized


def normalize_requested_malicious_fractions(values: Iterable[float]) -> list[float]:
    normalized = [round(float(value), 10) for value in values]
    invalid = sorted({value for value in normalized if all(abs(value - allowed) >= 1e-12 for allowed in SUPPORTED_MALICIOUS_FRACTIONS)})
    if invalid:
        rendered = ", ".join(str(value) for value in invalid)
        raise SystemExit(f"Unsupported malicious fraction(s): {rendered}")
    return normalized


def output_dir_is_new_enough(output_dir: Path, min_date: str = MIN_VALID_OUTPUT_DATE) -> bool:
    candidates = [output_dir]
    try:
        resolved = output_dir.resolve()
    except Exception:
        resolved = output_dir
    if resolved != output_dir:
        candidates.append(resolved)
    for candidate in candidates:
        timestamp_prefix = candidate.name[:8]
        if len(timestamp_prefix) == 8 and timestamp_prefix.isdigit() and timestamp_prefix >= str(min_date):
            return True
    return False


def fraction_tag(malicious_fraction: float) -> str:
    return f"y{int(round(float(malicious_fraction) * 10)):02d}"


def configure_partition(
    cfg: dict,
    *,
    spec_slug: str,
    partition: str,
    seed: int,
    node_defense_root: Path,
    metadata_absolute: bool,
) -> str:
    partition = canonical_partition_name(partition)
    cfg["federated"]["partition"] = partition
    cfg["federated"]["client_fraction"] = 1.0
    cfg["federated"]["min_client_size"] = 50

    if partition == "louvain":
        cfg["federated"]["alpha"] = 0.3
        cfg["federated"].pop("num_feature_clusters", None)
        partition_file = f"{spec_slug}_node_louvain_alpha03_seed{seed}_clients10_train60_val20_labeledv1.json"
    elif partition == "label_skew":
        cfg["federated"]["alpha"] = 0.5
        cfg["federated"].pop("num_feature_clusters", None)
        partition_file = f"{spec_slug}_node_label_skew_alpha05_seed{seed}_clients10_train60_val20_labeledv1.json"
    elif partition == "feature_skew":
        cfg["federated"]["alpha"] = 0.5
        cfg["federated"]["num_feature_clusters"] = int(cfg["federated"]["num_clients"])
        partition_file = (
            f"{spec_slug}_node_feature_skew_alpha05_clusters{cfg['federated']['num_feature_clusters']}_"
            f"seed{seed}_clients10_train60_val20_labeledv1.json"
        )
    elif partition == "iid":
        cfg["federated"].pop("alpha", None)
        cfg["federated"].pop("num_feature_clusters", None)
        partition_file = f"{spec_slug}_node_iid_seed{seed}_clients10_train60_val20_labeledv1.json"
    else:
        raise ValueError(f"Unsupported partition: {partition}")

    if metadata_absolute:
        partition_path = (node_defense_root / "partitions" / partition_file).resolve()
    else:
        partition_path = Path("partitions") / partition_file
    cfg["federated"]["partition_metadata_path"] = str(partition_path)
    return partition


def configure_trigger(cfg: dict, trigger: str) -> str:
    trigger = canonical_trigger_name(trigger)
    attack_cfg = cfg["attack"]
    attack_cfg["enabled"] = True
    attack_cfg["trigger_num_nodes"] = int(attack_cfg.get("trigger_num_nodes", 4))

    for key in (
        "ba_degree",
        "ws_degree",
        "ws_rewire_prob",
        "gta_trojan_epochs",
        "gta_hidden_dim",
        "gta_shadow_hidden_dim",
        "gta_edge_threshold",
        "gta_feature_reg_weight",
        "gta_edge_reg_weight",
    ):
        attack_cfg.pop(key, None)

    if trigger == "renyi":
        attack_cfg["name"] = "multi_client_renyi_trigger"
        attack_cfg["anchor_connection_mode"] = "all"
    elif trigger == "ba":
        attack_cfg["name"] = "multi_client_ba_trigger"
        attack_cfg["anchor_connection_mode"] = "single_root"
        attack_cfg["ba_degree"] = int(attack_cfg.get("ba_degree", max(1, min(2, attack_cfg["trigger_num_nodes"] - 1))))
    elif trigger == "ws":
        attack_cfg["name"] = "multi_client_ws_trigger"
        attack_cfg["anchor_connection_mode"] = "single_root"
        ws_degree = int(attack_cfg.get("ws_degree", max(2, min(4, attack_cfg["trigger_num_nodes"] - 1))))
        if ws_degree % 2 != 0:
            ws_degree = max(2, ws_degree - 1)
        attack_cfg["ws_degree"] = ws_degree
        attack_cfg["ws_rewire_prob"] = float(attack_cfg.get("ws_rewire_prob", 0.5))
    elif trigger == "gta":
        attack_cfg["name"] = "multi_client_gta_trigger"
        attack_cfg["anchor_connection_mode"] = "single_root"
        attack_cfg["gta_trojan_epochs"] = int(attack_cfg.get("gta_trojan_epochs", 60))
        attack_cfg["gta_hidden_dim"] = int(attack_cfg.get("gta_hidden_dim", 128))
        attack_cfg["gta_shadow_hidden_dim"] = int(attack_cfg.get("gta_shadow_hidden_dim", 64))
        attack_cfg["gta_edge_threshold"] = float(attack_cfg.get("gta_edge_threshold", 0.5))
        attack_cfg["gta_feature_reg_weight"] = float(attack_cfg.get("gta_feature_reg_weight", 0.1))
        attack_cfg["gta_edge_reg_weight"] = float(attack_cfg.get("gta_edge_reg_weight", 0.05))
    else:
        raise ValueError(f"Unsupported trigger: {trigger}")
    return trigger
