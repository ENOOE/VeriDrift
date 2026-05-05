from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.attack import build_local_client_attack, build_single_eval_attack_graph
from src.data import (
    SPLIT_PROTOCOL,
    attach_client_split_metadata,
    apply_partition_metadata,
    build_cached_clean_local_subgraphs,
    build_partition_metadata,
    create_custom_masks,
    dirichlet_partition_nodes,
    feature_skew_partition_nodes,
    iid_partition_nodes,
    load_planetoid_dataset,
    louvain_partition_nodes,
)
from src.federated import (
    aggregate_state_dicts,
    train_local_model,
)
from src.rscc_certifier import ProxySupervisedTrustCertifier
from src.model import NodeGCN
from src.risk_credit import OnlineRiskCreditTracker
from src.utils import (
    ensure_dir,
    load_json,
    normalize_runtime_paths,
    project_root_from_file,
    read_yaml,
    resolve_device,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train federated node classification with configurable partition and Renyi backdoor.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-rounds", type=int, default=None, help="Optional smoke-test override for global rounds.")
    return parser.parse_args()


def build_model_from_cfg(cfg: Dict[str, object], dataset) -> NodeGCN:
    model_cfg = cfg["model"]
    return NodeGCN(
        input_dim=int(dataset.num_features),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(dataset.num_classes),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg.get("dropout", 0.5)),
        layer_norm_first=bool(model_cfg.get("layer_norm_first", True)),
        use_ln=bool(model_cfg.get("use_ln", True)),
    )


def build_output_dir(cfg: Dict[str, object]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = str(cfg["dataset"]["name"]).lower()
    partition_name = canonical_partition_name(str(cfg["federated"].get("partition", "louvain")))
    attack_name = str(cfg.get("attack", {}).get("name", "clean")).lower()
    attack_tag = attack_name
    if attack_name.startswith("multi_client_") and attack_name.endswith("_trigger"):
        attack_tag = attack_name[len("multi_client_") : -len("_trigger")]
    return ensure_dir(Path(cfg["logging"]["output_root"]) / f"{timestamp}_{dataset_name}_node_{partition_name}_{attack_tag}")


def resolve_partition_metadata_path(cfg: Dict[str, object]) -> Path | None:
    federated_cfg = cfg.get("federated", {})
    raw_path = federated_cfg.get("partition_metadata_path") if isinstance(federated_cfg, dict) else None
    if not raw_path:
        return None
    return Path(str(raw_path))


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


DEFAULT_EVAL_LAST_N_ROUNDS = 5
DATASET_EVAL_LAST_N_ROUNDS = {
    "coauthorphysics": 3,
    "coauthorph": 3,
}
EVALUATION_PROTOCOL = {
    "split_protocol": SPLIT_PROTOCOL,
    "acc": "ACC is the mean over clients of clean node-classification accuracy on each client's full idx_test.",
    "asr": "ASR is the mean over malicious clients of target-class prediction rate on idx_test nodes whose clean label is not target_label.",
    "transfer_asr": "Transfer ASR is the mean target-class prediction rate when the malicious trigger is applied to benign clients' idx_test non-target nodes.",
    "clean_eval_schedule": "Coauthor-Ph logs clean val/test ACC only in the final 3 rounds; the first five datasets log clean val/test ACC only in the final 5 rounds. final_metrics.json is always evaluated on the final selected model.",
    "triggered_eval_schedule": "Coauthor-Ph logs ASR and Transfer ASR only in the final 3 rounds; the first five datasets log ASR and Transfer ASR only in the final 5 rounds. final_metrics.json is always evaluated on the final selected model.",
    "dataset_eval_last_n_rounds": {
        "default": DEFAULT_EVAL_LAST_N_ROUNDS,
        "coauthorphysics": 3,
    },
}
ASR_EVAL_PROTOCOL = "full_client_idx_test_non_target_nodes_individual_copy_v2"
TRANSFER_ASR_PROTOCOL = "malicious_trigger_to_benign_idx_test_non_target_nodes_individual_copy_v2"


def canonical_dataset_eval_name(name: str | None) -> str:
    if name is None:
        return ""
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def evaluation_last_n_rounds_for_dataset(dataset_name: str | None) -> int:
    return int(DATASET_EVAL_LAST_N_ROUNDS.get(canonical_dataset_eval_name(dataset_name), DEFAULT_EVAL_LAST_N_ROUNDS))


def _should_evaluate_round(round_idx: int, total_rounds: int, last_n_rounds: int) -> bool:
    return int(round_idx) > max(0, int(total_rounds) - int(last_n_rounds))


def should_evaluate_clean_round(
    round_idx: int,
    total_rounds: int,
    dataset_name: str | None = None,
    last_n_rounds: int | None = None,
) -> bool:
    if last_n_rounds is None:
        last_n_rounds = evaluation_last_n_rounds_for_dataset(dataset_name)
    return _should_evaluate_round(round_idx=round_idx, total_rounds=total_rounds, last_n_rounds=int(last_n_rounds))


def should_evaluate_attack_round(
    round_idx: int,
    total_rounds: int,
    dataset_name: str | None = None,
    last_n_rounds: int | None = None,
) -> bool:
    if last_n_rounds is None:
        last_n_rounds = evaluation_last_n_rounds_for_dataset(dataset_name)
    return _should_evaluate_round(round_idx=round_idx, total_rounds=total_rounds, last_n_rounds=int(last_n_rounds))


def build_skipped_clean_metrics() -> Dict[str, float]:
    return {
        "val_loss": float("nan"),
        "val_accuracy": float("nan"),
        "test_loss": float("nan"),
        "test_accuracy": float("nan"),
        "clean_test_accuracy": float("nan"),
        "num_clean_test_nodes": 0,
    }


@torch.no_grad()
def evaluate_client_average_splits(
    model: NodeGCN,
    local_graphs: List[object],
    device: torch.device,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()

    val_losses: List[float] = []
    val_accs: List[float] = []
    test_losses: List[float] = []
    test_accs: List[float] = []
    clean_test_accs: List[float] = []
    test_counts: List[int] = []

    for graph in local_graphs:
        eval_graph = graph.to(device)
        logits = model(eval_graph.x, eval_graph.edge_index, getattr(eval_graph, "edge_weight", None))
        preds = logits.argmax(dim=-1)

        if int(eval_graph.val_mask.sum().item()) > 0:
            val_losses.append(float(F.cross_entropy(logits[eval_graph.val_mask], eval_graph.y[eval_graph.val_mask]).item()))
            val_accs.append(float((preds[eval_graph.val_mask] == eval_graph.y[eval_graph.val_mask]).float().mean().item()))

        if int(eval_graph.test_mask.sum().item()) > 0:
            test_losses.append(float(F.cross_entropy(logits[eval_graph.test_mask], eval_graph.y[eval_graph.test_mask]).item()))
            test_accs.append(float((preds[eval_graph.test_mask] == eval_graph.y[eval_graph.test_mask]).float().mean().item()))
            test_counts.append(int(eval_graph.test_mask.sum().item()))

        if hasattr(eval_graph, "clean_test_mask") and int(eval_graph.clean_test_mask.sum().item()) > 0:
            clean_test_accs.append(
                float((preds[eval_graph.clean_test_mask] == eval_graph.y[eval_graph.clean_test_mask]).float().mean().item())
            )

    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
        "val_accuracy": float(np.mean(val_accs)) if val_accs else float("nan"),
        "test_loss": float(np.mean(test_losses)) if test_losses else float("nan"),
        "test_accuracy": float(np.mean(test_accs)) if test_accs else float("nan"),
        "clean_test_accuracy": float(np.mean(clean_test_accs)) if clean_test_accs else float("nan"),
        "num_clean_test_nodes": int(sum(test_counts)),
    }


def _idx_atk_test(clean_graph: object, target_label: int) -> List[int]:
    test_nodes = torch.where(clean_graph.test_mask)[0].detach().cpu().tolist()
    return [
        int(node)
        for node in test_nodes
        if int(clean_graph.clean_y[int(node)].detach().cpu().item()) != int(target_label)
    ]


@torch.no_grad()
def _evaluate_triggered_nodes(
    model: NodeGCN,
    clean_graph: object,
    victim_nodes: List[int],
    attack_runtime: Dict[str, object],
    device: torch.device,
    seed: int,
    source_client_id: int | None = None,
) -> Dict[str, float]:
    if not victim_nodes:
        return {"loss": float("nan"), "success_rate": float("nan"), "num_eval_nodes": 0}
    target_label = int(attack_runtime["target_label"])
    losses: List[float] = []
    successes = 0
    for offset, victim_node in enumerate(victim_nodes):
        attacked_eval_graph = build_single_eval_attack_graph(
            clean_data=clean_graph,
            victim_node=int(victim_node),
            attack_runtime=attack_runtime,
            seed=seed + 13 * int(offset) + 1009 * int(victim_node),
            source_client_id=source_client_id,
        ).to(device)
        logits = model(
            attacked_eval_graph.x,
            attacked_eval_graph.edge_index,
            getattr(attacked_eval_graph, "edge_weight", None),
        )
        target = torch.tensor([target_label], dtype=torch.long, device=device)
        losses.append(float(F.cross_entropy(logits[[int(victim_node)]], target).item()))
        pred = int(logits[int(victim_node)].argmax(dim=-1).item())
        successes += int(pred == target_label)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "success_rate": float(successes / max(1, len(victim_nodes))),
        "num_eval_nodes": int(len(victim_nodes)),
    }


@torch.no_grad()
def evaluate_local_asr(
    model: NodeGCN,
    clean_local_graphs: List[object],
    malicious_client_ids: List[int],
    attack_runtime: Dict[str, object],
    device: torch.device,
    eval_seed: int,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()
    target_label = int(attack_runtime["target_label"])

    losses: List[float] = []
    asr_values: List[float] = []
    total_eval_nodes = 0
    per_client_nodes: Dict[str, int] = {}
    per_client_asr: Dict[str, float] = {}

    for client_id in malicious_client_ids:
        clean_graph = clean_local_graphs[int(client_id)]
        flip_nodes = _idx_atk_test(clean_graph, target_label)
        if not flip_nodes:
            continue

        client_metrics = _evaluate_triggered_nodes(
            model=model,
            clean_graph=clean_graph,
            victim_nodes=flip_nodes,
            attack_runtime=attack_runtime,
            device=device,
            seed=eval_seed + 1000 * int(client_id),
            source_client_id=int(client_id),
        )
        losses.append(float(client_metrics["loss"]))
        asr_values.append(float(client_metrics["success_rate"]))
        total_eval_nodes += int(client_metrics["num_eval_nodes"])
        per_client_nodes[str(int(client_id))] = int(client_metrics["num_eval_nodes"])
        per_client_asr[str(int(client_id))] = float(client_metrics["success_rate"])

    return {
        "attack_loss": float(np.mean(losses)) if losses else float("nan"),
        "asr": float(np.mean(asr_values)) if asr_values else float("nan"),
        "num_eval_nodes": int(total_eval_nodes),
        "per_client_eval_nodes": per_client_nodes,
        "per_client_asr": per_client_asr,
        "eval_protocol": ASR_EVAL_PROTOCOL,
    }


@torch.no_grad()
def evaluate_transfer_asr(
    model: NodeGCN,
    clean_local_graphs: List[object],
    malicious_client_ids: List[int],
    attack_runtime: Dict[str, object],
    device: torch.device,
    eval_seed: int,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()
    target_label = int(attack_runtime["target_label"])
    malicious_set = {int(client_id) for client_id in malicious_client_ids}
    benign_client_ids = [idx for idx in range(len(clean_local_graphs)) if idx not in malicious_set]

    losses: List[float] = []
    transfer_values: List[float] = []
    total_eval_nodes = 0
    per_client_nodes: Dict[str, int] = {}
    per_client_asr: Dict[str, List[float]] = {}

    per_pair_asr: Dict[str, float] = {}
    per_pair_nodes: Dict[str, int] = {}
    for source_client_id in sorted(malicious_set):
        for client_id in benign_client_ids:
            clean_graph = clean_local_graphs[int(client_id)]
            flip_nodes = _idx_atk_test(clean_graph, target_label)
            if not flip_nodes:
                continue
            pair_metrics = _evaluate_triggered_nodes(
                model=model,
                clean_graph=clean_graph,
                victim_nodes=flip_nodes,
                attack_runtime=attack_runtime,
                device=device,
                seed=eval_seed + 7000 * int(source_client_id) + 101 * int(client_id),
                source_client_id=int(source_client_id),
            )
            losses.append(float(pair_metrics["loss"]))
            transfer_values.append(float(pair_metrics["success_rate"]))
            total_eval_nodes += int(pair_metrics["num_eval_nodes"])
            pair_key = f"{int(source_client_id)}->{int(client_id)}"
            per_pair_nodes[pair_key] = int(pair_metrics["num_eval_nodes"])
            per_pair_asr[pair_key] = float(pair_metrics["success_rate"])
            per_client_nodes[str(int(client_id))] = per_client_nodes.get(str(int(client_id)), 0) + int(
                pair_metrics["num_eval_nodes"]
            )
            per_client_asr.setdefault(str(int(client_id)), []).append(float(pair_metrics["success_rate"]))

    per_client_asr_mean = {
        client_id: float(np.mean(values))
        for client_id, values in per_client_asr.items()
    }

    return {
        "transfer_attack_loss": float(np.mean(losses)) if losses else float("nan"),
        "transfer_asr": float(np.mean(transfer_values)) if transfer_values else float("nan"),
        "num_transfer_eval_nodes": int(total_eval_nodes),
        "per_benign_client_eval_nodes": per_client_nodes,
        "per_benign_client_transfer_asr": per_client_asr_mean,
        "num_transfer_sources": int(len(malicious_set)),
        "per_source_benign_pair_transfer_asr": per_pair_asr,
        "per_source_benign_pair_eval_nodes": per_pair_nodes,
        "eval_protocol": TRANSFER_ASR_PROTOCOL,
    }


def populate_final_attack_metrics(
    final_payload: Dict[str, object],
    *,
    model: NodeGCN,
    clean_local_graphs: List[object],
    malicious_set: set[int],
    attack_runtime: Dict[str, object] | None,
    attack_metadata: Dict[str, object] | None,
    attack_start_round: int,
    device: torch.device,
    asr_eval_seed: int,
    total_rounds: int,
    last_attack_eval_round: int,
    last_attack_metrics: Dict[str, object] | None,
    last_transfer_metrics: Dict[str, object] | None,
) -> None:
    selected_round = int(final_payload["selected_round"])
    if attack_runtime is None or attack_metadata is None or selected_round < int(attack_start_round):
        return

    if (
        last_attack_metrics is None
        or last_transfer_metrics is None
        or int(last_attack_eval_round) < int(attack_start_round)
    ):
        return

    attack_metrics = dict(last_attack_metrics)
    transfer_metrics = dict(last_transfer_metrics)

    final_payload["attack_loss"] = attack_metrics["attack_loss"]
    final_payload["asr"] = attack_metrics["asr"]
    final_payload["num_attack_eval_nodes"] = int(attack_metrics["num_eval_nodes"])
    final_payload["per_client_asr"] = attack_metrics["per_client_asr"]
    final_payload["per_client_attack_eval_nodes"] = attack_metrics["per_client_eval_nodes"]
    final_payload["transfer_attack_loss"] = transfer_metrics["transfer_attack_loss"]
    final_payload["transfer_asr"] = transfer_metrics["transfer_asr"]
    final_payload["num_transfer_eval_nodes"] = int(transfer_metrics["num_transfer_eval_nodes"])
    final_payload["per_benign_client_transfer_asr"] = transfer_metrics["per_benign_client_transfer_asr"]
    final_payload["final_attack_metrics_source"] = "reused_cached_attack_eval"
    final_payload["final_attack_metrics_round"] = int(last_attack_eval_round)
    final_payload["per_benign_client_transfer_eval_nodes"] = transfer_metrics["per_benign_client_eval_nodes"]


def main() -> None:
    args = parse_args()
    cfg = normalize_runtime_paths(
        read_yaml(args.config),
        project_root=project_root_from_file(__file__),
    )
    if args.max_rounds is not None:
        cfg["training"] = dict(cfg["training"])
        cfg["training"]["global_rounds"] = int(args.max_rounds)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(args.device)

    dataset = load_planetoid_dataset(
        root=cfg["dataset"]["root"],
        name=str(cfg["dataset"]["name"]),
    )
    full_data = dataset[0]
    full_data = create_custom_masks(
        data=full_data,
        train_ratio=float(cfg["dataset"]["train_ratio"]),
        val_ratio=float(cfg["dataset"]["val_ratio"]),
        seed=int(cfg["seed"]),
    )
    remaining_data = full_data
    if not hasattr(remaining_data, "index_orig"):
        remaining_data.index_orig = torch.arange(remaining_data.num_nodes, dtype=torch.long)

    partition_name = canonical_partition_name(str(cfg["federated"].get("partition", "louvain")))
    community_lists = None
    feature_cluster_map = None
    partition_metadata_path = resolve_partition_metadata_path(cfg)
    if partition_metadata_path is not None and partition_metadata_path.exists():
        partition_metadata = load_json(partition_metadata_path)
        cached_partition_name = canonical_partition_name(str(partition_metadata.get("partition_name", "")))
        if cached_partition_name != partition_name:
            raise ValueError(
                f"Cached partition {partition_metadata_path} is for "
                f"{partition_metadata.get('partition_name')}, expected {partition_name}."
            )
        if int(partition_metadata.get("num_nodes", -1)) != int(remaining_data.num_nodes):
            raise ValueError(
                f"Cached partition {partition_metadata_path} has "
                f"{partition_metadata.get('num_nodes')} nodes, expected {remaining_data.num_nodes}."
            )
        if len(partition_metadata.get("clients", [])) != int(cfg["federated"]["num_clients"]):
            raise ValueError(
                f"Cached partition {partition_metadata_path} has "
                f"{len(partition_metadata.get('clients', []))} clients, expected {cfg['federated']['num_clients']}."
            )
        remaining_data = apply_partition_metadata(remaining_data, partition_metadata)
        full_data = remaining_data
        client_node_indices = [
            [int(node) for node in client_info["train_node_indices"]]
            for client_info in partition_metadata["clients"]
        ]
        if partition_name == "feature_skew":
            feature_cluster_map = {
                int(item["node_index"]): int(item["cluster_id"])
                for item in partition_metadata.get("feature_clusters", [])
            }
    else:
        if partition_name == "louvain":
            client_node_indices, community_lists = louvain_partition_nodes(
                data=remaining_data,
                num_clients=int(cfg["federated"]["num_clients"]),
                seed=int(cfg["seed"]),
                min_size=int(cfg["federated"].get("min_client_size", 20)),
            )
        elif partition_name == "label_skew":
            train_indices = torch.where(remaining_data.train_mask)[0]
            client_node_indices = dirichlet_partition_nodes(
                labels=remaining_data.y,
                train_indices=train_indices,
                num_clients=int(cfg["federated"]["num_clients"]),
                alpha=float(cfg["federated"]["alpha"]),
                seed=int(cfg["seed"]),
                min_size=int(cfg["federated"].get("min_client_size", 20)),
            )
        elif partition_name == "feature_skew":
            train_indices = torch.where(remaining_data.train_mask)[0]
            client_node_indices, feature_cluster_map = feature_skew_partition_nodes(
                features=remaining_data.x,
                train_indices=train_indices,
                num_clients=int(cfg["federated"]["num_clients"]),
                alpha=float(cfg["federated"]["alpha"]),
                seed=int(cfg["seed"]),
                min_size=int(cfg["federated"].get("min_client_size", 20)),
                num_feature_clusters=cfg["federated"].get("num_feature_clusters"),
            )
        elif partition_name == "iid":
            train_indices = torch.where(remaining_data.train_mask)[0]
            client_node_indices = iid_partition_nodes(
                labels=remaining_data.y,
                train_indices=train_indices,
                num_clients=int(cfg["federated"]["num_clients"]),
                seed=int(cfg["seed"]),
            )
        else:
            raise ValueError(f"Unsupported partition: {partition_name}")

        partition_metadata = build_partition_metadata(
            data=remaining_data,
            client_node_indices=client_node_indices,
            partition_name=partition_name,
            community_lists=community_lists,
            feature_cluster_map=feature_cluster_map,
            partition_params={
                key: cfg["federated"][key]
                for key in ("alpha", "num_feature_clusters")
                if key in cfg["federated"]
            },
        )
        if partition_metadata_path is not None:
            save_json(partition_metadata, partition_metadata_path)

    clean_local_graphs = build_cached_clean_local_subgraphs(
        source_data=remaining_data,
        client_node_indices=client_node_indices,
        full_data=full_data,
        seed=int(cfg["seed"]),
        partition_metadata_path=partition_metadata_path,
        partition_name=partition_name,
    )

    attack_metadata = None
    attack_runtime = None
    attacked_local_graphs = clean_local_graphs
    if bool(cfg.get("attack", {}).get("enabled", False)):
        attacked_local_graphs, attack_metadata, attack_runtime = build_local_client_attack(
            local_graphs=clean_local_graphs,
            attack_cfg=cfg["attack"],
            seed=int(cfg["seed"]),
        )

    for client_info in partition_metadata["clients"]:
        client_info["is_malicious"] = False
    malicious_set = set()
    if attack_metadata is not None:
        malicious_set = set(int(client_id) for client_id in attack_metadata["malicious_client_ids"])
        for client_info in partition_metadata["clients"]:
            client_info["is_malicious"] = int(client_info["client_id"]) in malicious_set
        if attack_runtime is not None:
            attack_runtime["eval_protocol"] = ASR_EVAL_PROTOCOL
            attack_runtime["transfer_eval_protocol"] = TRANSFER_ASR_PROTOCOL
            attack_metadata["asr_eval_protocol"] = ASR_EVAL_PROTOCOL
            attack_metadata["transfer_asr_eval_protocol"] = TRANSFER_ASR_PROTOCOL

    partition_metadata = attach_client_split_metadata(
        partition_metadata=partition_metadata,
        local_graphs=clean_local_graphs,
        target_label=int(attack_runtime["target_label"]) if attack_runtime is not None else None,
    )
    partition_metadata["evaluation_protocol"] = EVALUATION_PROTOCOL

    output_dir = build_output_dir(cfg)
    ensure_dir(output_dir / "checkpoints")
    risk_credit_dir = ensure_dir(output_dir / "risk_credit")
    server_certifier_dir = ensure_dir(output_dir / "server_certifier")
    save_json(cfg, output_dir / "config.json")
    save_json(partition_metadata, output_dir / "partition_metadata.json")
    if attack_metadata is not None:
        save_json(attack_metadata, output_dir / "attack_metadata.json")

    risk_credit_tracker = OnlineRiskCreditTracker(
        output_dir=risk_credit_dir,
        malicious_client_ids=sorted(malicious_set),
    )
    certifier_cfg = dict(cfg.get("trust_certifier", {}))
    server_certifier = ProxySupervisedTrustCertifier(
        output_dir=server_certifier_dir,
        malicious_client_ids=sorted(malicious_set),
        device=str(device),
        risk_alpha=float(certifier_cfg.get("risk_alpha", 10.0)),
        consistency_bonus=float(certifier_cfg.get("consistency_bonus", 0.5)),
        warmup_tau=float(certifier_cfg.get("warmup_tau", 20.0)),
        risk_memory_decay=float(certifier_cfg.get("risk_memory_decay", 0.90)),
    )

    model = build_model_from_cfg(cfg, dataset)
    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "round",
        "attack_active",
        "effective_poison_loss_weight",
        "val_loss",
        "val_accuracy",
        "test_loss",
        "test_accuracy",
        "mean_local_loss",
        "attack_loss",
        "asr",
        "transfer_attack_loss",
        "transfer_asr",
        "num_attack_eval_nodes",
        "num_transfer_eval_nodes",
        "num_malicious_clients",
        "num_poisoned_nodes",
        "cert_stage",
        "num_trusted",
        "num_suspicious",
        "num_reject",
        "accepted_suspicious_weight_ratio",
        "rejected_client_ratio",
        "mean_suspicious_trust_probability",
        "mlp_trained",
        "mlp_loss",
        "selected_as_best",
    ]

    poison_loss_weight = float(cfg.get("attack", {}).get("poison_loss_weight", 1.0))
    malicious_local_epochs = int(cfg.get("attack", {}).get("malicious_local_epochs", int(cfg["training"]["local_epochs"])))
    attack_start_round = int(cfg.get("attack", {}).get("start_round", 1))
    attack_ramp_rounds = max(1, int(cfg.get("attack", {}).get("ramp_rounds", 1)))
    restore_best_checkpoint = bool(cfg.get("logging", {}).get("restore_best_checkpoint", True))
    asr_eval_seed = int(cfg["seed"]) + 100000

    best_state_dict = None
    best_round = 0
    best_val_accuracy = float("-inf")
    best_selection_score = float("-inf")
    best_metrics: Dict[str, object] | None = None
    last_attack_metrics: Dict[str, object] | None = None
    last_transfer_metrics: Dict[str, object] | None = None
    last_attack_eval_round = 0

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()

        for round_idx in range(1, int(cfg["training"]["global_rounds"]) + 1):
            attack_active = attack_runtime is not None and round_idx >= attack_start_round
            if attack_active:
                ramp_progress = min(1.0, float(round_idx - attack_start_round + 1) / float(attack_ramp_rounds))
                effective_poison_loss_weight = 1.0 + ramp_progress * (poison_loss_weight - 1.0)
            else:
                effective_poison_loss_weight = 1.0

            local_results = []
            active_local_graphs = attacked_local_graphs if attack_active else clean_local_graphs
            for client_id, local_graph in enumerate(active_local_graphs):
                is_malicious = attack_active and client_id in malicious_set
                local_result = train_local_model(
                    global_model=model,
                    local_data=local_graph,
                    owned_train_mask=getattr(local_graph, "owned_train_mask", local_graph.train_mask),
                    lr=float(cfg["training"]["lr"]),
                    weight_decay=float(cfg["training"]["weight_decay"]),
                    local_epochs=malicious_local_epochs if is_malicious else int(cfg["training"]["local_epochs"]),
                    device=device,
                    poison_loss_weight=effective_poison_loss_weight if is_malicious else 1.0,
                    return_risk_payload=True,
                )
                local_results.append(local_result)

            risk_credit_tracker.update_from_local_results(round_idx=round_idx, local_results=local_results)
            certifier_summary = server_certifier.certify_round(
                round_idx=round_idx,
                global_model=model,
                global_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                local_results=local_results,
                client_risk_rows=risk_credit_tracker.get_latest_client_rows(),
            )
            latest_weights = server_certifier.get_latest_weights()
            aggregated_state = aggregate_state_dicts(
                state_dicts=[item["state_dict"] for item in local_results],
                weights=latest_weights,
            )

            model.load_state_dict(aggregated_state)

            clean_eval_active = should_evaluate_clean_round(
                round_idx=round_idx,
                total_rounds=int(cfg["training"]["global_rounds"]),
                dataset_name=str(cfg["dataset"]["name"]),
            )
            clean_metrics = (
                evaluate_client_average_splits(model=model, local_graphs=clean_local_graphs, device=device)
                if clean_eval_active
                else build_skipped_clean_metrics()
            )
            row = {
                "round": round_idx,
                "attack_active": int(attack_active),
                "effective_poison_loss_weight": float(effective_poison_loss_weight),
                "val_loss": clean_metrics["val_loss"] if clean_eval_active else "",
                "val_accuracy": clean_metrics["val_accuracy"] if clean_eval_active else "",
                "test_loss": clean_metrics["test_loss"] if clean_eval_active else "",
                "test_accuracy": clean_metrics["test_accuracy"] if clean_eval_active else "",
                "mean_local_loss": float(sum(item["mean_loss"] for item in local_results) / len(local_results)),
                "attack_loss": "",
                "asr": "",
                "transfer_attack_loss": "",
                "transfer_asr": "",
                "num_attack_eval_nodes": 0,
                "num_transfer_eval_nodes": 0,
                "num_malicious_clients": 0 if attack_metadata is None else int(attack_metadata["num_malicious_clients"]),
                "num_poisoned_nodes": 0 if attack_metadata is None else int(len(attack_metadata["all_poisoned_nodes"])),
                "cert_stage": str(certifier_summary["stage"]),
                "num_trusted": int(certifier_summary["num_trusted"]),
                "num_suspicious": int(certifier_summary["num_suspicious"]),
                "num_reject": int(certifier_summary["num_reject"]),
                "accepted_suspicious_weight_ratio": float(certifier_summary["accepted_suspicious_weight_ratio"]),
                "rejected_client_ratio": float(certifier_summary["rejected_client_ratio"]),
                "mean_suspicious_trust_probability": float(certifier_summary["mean_suspicious_trust_probability"]),
                "mlp_trained": float(certifier_summary["mlp_trained"]),
                "mlp_loss": float(certifier_summary["mlp_loss"]),
                "selected_as_best": 0,
            }

            attack_metrics = None
            if (
                attack_active
                and attack_runtime is not None
                and attack_metadata is not None
                and should_evaluate_attack_round(
                    round_idx=round_idx,
                    total_rounds=int(cfg["training"]["global_rounds"]),
                    dataset_name=str(cfg["dataset"]["name"]),
                )
            ):
                attack_metrics = evaluate_local_asr(
                    model=model,
                    clean_local_graphs=clean_local_graphs,
                    malicious_client_ids=sorted(malicious_set),
                    attack_runtime=attack_runtime,
                    device=device,
                    eval_seed=asr_eval_seed,
                )
                row["attack_loss"] = attack_metrics["attack_loss"]
                row["asr"] = attack_metrics["asr"]
                row["num_attack_eval_nodes"] = int(attack_metrics["num_eval_nodes"])
                transfer_metrics = evaluate_transfer_asr(
                    model=model,
                    clean_local_graphs=clean_local_graphs,
                    malicious_client_ids=sorted(malicious_set),
                    attack_runtime=attack_runtime,
                    device=device,
                    eval_seed=asr_eval_seed + 200000,
                )
                row["transfer_attack_loss"] = transfer_metrics["transfer_attack_loss"]
                row["transfer_asr"] = transfer_metrics["transfer_asr"]
                row["num_transfer_eval_nodes"] = int(transfer_metrics["num_transfer_eval_nodes"])
                last_attack_metrics = dict(attack_metrics)
                last_transfer_metrics = dict(transfer_metrics)
                last_attack_eval_round = int(round_idx)

            if clean_eval_active:
                selection_score = float(clean_metrics["val_accuracy"])
                if attack_metrics is not None:
                    selection_score -= float(attack_metrics["asr"])
                if (
                    best_metrics is None
                    or selection_score > best_selection_score
                    or (
                        selection_score == best_selection_score
                        and clean_metrics["test_accuracy"] > float(best_metrics["test_accuracy"])
                    )
                ):
                    best_selection_score = float(selection_score)
                    best_val_accuracy = clean_metrics["val_accuracy"]
                    best_round = round_idx
                    best_state_dict = deepcopy(model.state_dict())
                    row["selected_as_best"] = 1
                    best_metrics = dict(row)

            writer.writerow(row)
            handle.flush()

            if round_idx % int(cfg["logging"]["save_model_every"]) == 0:
                torch.save(
                    {"round": round_idx, "model_state_dict": model.state_dict()},
                    output_dir / "checkpoints" / f"round_{round_idx:03d}.pt",
                )

    if restore_best_checkpoint and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(
            {"round": best_round, "model_state_dict": best_state_dict},
            output_dir / "checkpoints" / "best_model.pt",
        )

    final_metrics = evaluate_client_average_splits(model=model, local_graphs=clean_local_graphs, device=device)
    reported_round = best_round if restore_best_checkpoint and best_round else int(cfg["training"]["global_rounds"])
    final_payload = {
        "selected_round": int(reported_round),
        "best_round": int(best_round if best_round else int(cfg["training"]["global_rounds"])),
        "restored_best_checkpoint": bool(restore_best_checkpoint),
        "best_val_accuracy": float(best_val_accuracy if best_metrics is not None else final_metrics["val_accuracy"]),
        "val_loss": final_metrics["val_loss"],
        "val_accuracy": final_metrics["val_accuracy"],
        "test_loss": final_metrics["test_loss"],
        "test_accuracy": final_metrics["test_accuracy"],
        "clean_test_accuracy": final_metrics["clean_test_accuracy"],
        "mean_local_loss": float(best_metrics["mean_local_loss"]) if best_metrics is not None else float("nan"),
        "attack_loss": "",
        "asr": "",
        "transfer_attack_loss": "",
        "transfer_asr": "",
        "num_attack_eval_nodes": 0,
        "num_transfer_eval_nodes": 0,
        "num_malicious_clients": 0 if attack_metadata is None else int(attack_metadata["num_malicious_clients"]),
        "num_poisoned_nodes": 0 if attack_metadata is None else int(len(attack_metadata["all_poisoned_nodes"])),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "asr_eval_protocol": ASR_EVAL_PROTOCOL,
        "transfer_asr_eval_protocol": TRANSFER_ASR_PROTOCOL,
    }
    populate_final_attack_metrics(
        final_payload,
        model=model,
        clean_local_graphs=clean_local_graphs,
        malicious_set=malicious_set,
        attack_runtime=attack_runtime,
        attack_metadata=attack_metadata,
        attack_start_round=attack_start_round,
        device=device,
        asr_eval_seed=asr_eval_seed,
        total_rounds=int(cfg["training"]["global_rounds"]),
        last_attack_eval_round=last_attack_eval_round,
        last_attack_metrics=last_attack_metrics,
        last_transfer_metrics=last_transfer_metrics,
    )

    save_json(final_payload, output_dir / "final_metrics.json")
    risk_credit_summary = risk_credit_tracker.finalize()
    if risk_credit_summary:
        save_json(risk_credit_summary, risk_credit_dir / "risk_credit_summary.json")
    certifier_summary = server_certifier.finalize()
    if certifier_summary:
        save_json(certifier_summary, server_certifier_dir / "server_certifier_summary.json")
    print(f"Training finished. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
