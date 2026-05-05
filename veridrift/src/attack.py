from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.data import refresh_split_indices
from src.model import NodeGCN


def build_renyi_trigger_edges(
    trigger_num_nodes: int,
    edge_prob: float,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    edges: List[Tuple[int, int]] = []
    for src in range(trigger_num_nodes):
        for dst in range(src + 1, trigger_num_nodes):
            if rng.random() < edge_prob:
                edges.append((src, dst))
    if not edges:
        chain_edges = [(idx, idx + 1) for idx in range(trigger_num_nodes - 1)]
        edges.extend(chain_edges[:1] if chain_edges else [])
    return edges


def randomize_trigger_edges(
    edges: Sequence[Tuple[int, int]],
    trigger_num_nodes: int,
    add_prob: float,
    drop_prob: float,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    edge_set = {tuple(sorted(edge)) for edge in edges}
    retained = set()
    for edge in edge_set:
        if rng.random() >= drop_prob:
            retained.add(edge)

    all_pairs = {
        (src, dst)
        for src in range(trigger_num_nodes)
        for dst in range(src + 1, trigger_num_nodes)
    }
    for edge in sorted(all_pairs - retained):
        if rng.random() < add_prob:
            retained.add(edge)

    if not retained:
        retained.add((0, 1))
    return sorted(retained)


def build_watts_strogatz_trigger_edges(
    trigger_num_nodes: int,
    degree: int,
    rewire_prob: float,
    seed: int,
) -> List[Tuple[int, int]]:
    if trigger_num_nodes <= 1:
        return []
    degree = int(max(2, degree))
    degree = min(degree, max(1, trigger_num_nodes - 1))
    if degree % 2 != 0:
        degree = max(2, degree - 1)
    graph = nx.watts_strogatz_graph(
        n=int(trigger_num_nodes),
        k=int(degree),
        p=float(np.clip(rewire_prob, 0.0, 1.0)),
        seed=int(seed),
    )
    edges = sorted((int(min(src, dst)), int(max(src, dst))) for src, dst in graph.edges())
    if not edges and trigger_num_nodes >= 2:
        edges = [(0, 1)]
    return edges


def build_barabasi_albert_trigger_edges(
    trigger_num_nodes: int,
    degree: int,
    seed: int,
) -> List[Tuple[int, int]]:
    if trigger_num_nodes <= 1:
        return []
    degree = int(max(1, degree))
    degree = min(degree, max(1, trigger_num_nodes - 1))
    graph = nx.barabasi_albert_graph(
        n=int(trigger_num_nodes),
        m=int(degree),
        seed=int(seed),
    )
    edges = sorted((int(min(src, dst)), int(max(src, dst))) for src, dst in graph.edges())
    if not edges and trigger_num_nodes >= 2:
        edges = [(0, 1)]
    return edges


def _complete_trigger_edge_template(trigger_num_nodes: int) -> List[Tuple[int, int]]:
    return [
        (src, dst)
        for src in range(int(trigger_num_nodes))
        for dst in range(src + 1, int(trigger_num_nodes))
    ]


def canonical_attack_name(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "renyi": "multi_client_renyi_trigger",
        "multi_client_renyi_trigger": "multi_client_renyi_trigger",
        "ba": "multi_client_ba_trigger",
        "multi_client_ba_trigger": "multi_client_ba_trigger",
        "ws": "multi_client_ws_trigger",
        "multi_client_ws_trigger": "multi_client_ws_trigger",
        "gta": "multi_client_gta_trigger",
        "multi_client_gta_trigger": "multi_client_gta_trigger",
    }
    return aliases.get(key, key)


def attack_tag_from_name(name: str) -> str:
    canonical = canonical_attack_name(name)
    if canonical.startswith("multi_client_") and canonical.endswith("_trigger"):
        return canonical[len("multi_client_") : -len("_trigger")]
    return canonical


class GraphTrojanNet(nn.Module):
    def __init__(self, input_dim: int, trigger_num_nodes: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.trigger_num_nodes = int(trigger_num_nodes)
        self.input_dim = int(input_dim)
        self.pair_dim = max(0, self.trigger_num_nodes * (self.trigger_num_nodes - 1) // 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.feat_head = nn.Linear(int(hidden_dim), self.trigger_num_nodes * self.input_dim)
        self.edge_head = nn.Linear(int(hidden_dim), self.pair_dim) if self.pair_dim > 0 else None

    def forward(self, victim_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(victim_features)
        trigger_features = self.feat_head(hidden).view(-1, self.trigger_num_nodes, self.input_dim)
        if self.edge_head is None:
            edge_logits = torch.empty(victim_features.size(0), 0, device=victim_features.device, dtype=victim_features.dtype)
        else:
            edge_logits = self.edge_head(hidden)
        return trigger_features, edge_logits


def build_target_feature_prototype(clean_data: Data, target_label: int) -> torch.Tensor:
    mask = (clean_data.clean_y == int(target_label)) & clean_data.train_mask
    if int(mask.sum().item()) == 0:
        mask = clean_data.clean_y == int(target_label)
    if int(mask.sum().item()) == 0:
        return clean_data.x.float().mean(dim=0)
    return clean_data.x[mask].float().mean(dim=0)


def choose_attack_target_label(clean_data: Data, preferred_target: int) -> int:
    labels = clean_data.clean_y[clean_data.train_mask].cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    label_count = {int(label): int(count) for label, count in zip(unique, counts)}
    if int(preferred_target) in label_count:
        return int(preferred_target)
    return min(label_count, key=label_count.get)


def choose_attack_target_label_from_graphs(local_graphs: Sequence[Data], preferred_target: int) -> int:
    all_labels: List[int] = []
    for graph in local_graphs:
        if hasattr(graph, "train_mask"):
            all_labels.extend(graph.clean_y[graph.train_mask].detach().cpu().tolist())
        else:
            all_labels.extend(graph.clean_y.detach().cpu().tolist())
    if not all_labels:
        return int(preferred_target)
    unique, counts = np.unique(np.array(all_labels), return_counts=True)
    label_count = {int(label): int(count) for label, count in zip(unique, counts)}
    if int(preferred_target) in label_count:
        return int(preferred_target)
    return min(label_count, key=label_count.get)


def select_malicious_clients(
    client_node_indices: Sequence[Sequence[int]],
    malicious_fraction: float,
    seed: int,
) -> List[int]:
    num_clients = len(client_node_indices)
    num_malicious = max(1, int(round(num_clients * malicious_fraction)))
    ranked = sorted(range(num_clients), key=lambda idx: len(client_node_indices[idx]), reverse=True)
    rng = np.random.default_rng(seed)
    top_pool = ranked[: max(num_malicious, min(num_clients, num_malicious + 2))]
    selected = sorted(rng.choice(top_pool, size=num_malicious, replace=False).tolist())
    return [int(client_id) for client_id in selected]


def _build_unlabeled_neighbor_pool(
    clean_data: Data,
    owned_train_nodes: Sequence[int],
    target_label: int,
) -> List[int]:
    split_mask = clean_data.train_mask | clean_data.val_mask | clean_data.test_mask
    unlabeled_mask = getattr(clean_data, "unlabeled_mask", torch.bitwise_not(split_mask))
    unlabeled_mask = unlabeled_mask & torch.bitwise_not(split_mask)
    non_target_mask = clean_data.clean_y != int(target_label)
    candidate_mask = unlabeled_mask & non_target_mask

    if int(candidate_mask.sum().item()) == 0:
        return []

    edge_index = clean_data.edge_index.detach().cpu()
    owned_set = {int(node) for node in owned_train_nodes}
    neighbor_pool: Set[int] = set()
    for src, dst in edge_index.t().tolist():
        if int(src) in owned_set and bool(candidate_mask[int(dst)].item()):
            neighbor_pool.add(int(dst))
        if int(dst) in owned_set and bool(candidate_mask[int(src)].item()):
            neighbor_pool.add(int(src))
    return sorted(neighbor_pool)


def _select_attach_nodes_local(
    local_graph: Data,
    target_label: int,
    poison_rate: float,
    seed: int,
) -> List[int]:
    split_mask = local_graph.train_mask | local_graph.val_mask | local_graph.test_mask
    unlabeled_mask = getattr(local_graph, "unlabeled_mask", torch.bitwise_not(split_mask))
    unlabeled_idx = torch.where(unlabeled_mask & torch.bitwise_not(split_mask))[0].tolist()
    eligible = [int(node) for node in unlabeled_idx if int(local_graph.clean_y[int(node)].item()) != int(target_label)]
    if not eligible:
        return []
    desired = max(1, int(np.floor(int(local_graph.train_mask.sum().item()) * poison_rate)))
    desired = min(desired, len(eligible))
    if desired <= 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=desired, replace=False).tolist())


def select_poisoned_nodes_for_client(
    clean_data: Data,
    owned_train_nodes: Sequence[int],
    target_label: int,
    poison_rate: float,
    seed: int,
    used_nodes: Set[int] | None = None,
) -> List[int]:
    if used_nodes is None:
        used_nodes = set()

    unlabeled_pool = _build_unlabeled_neighbor_pool(
        clean_data=clean_data,
        owned_train_nodes=owned_train_nodes,
        target_label=target_label,
    )
    eligible = [int(node) for node in unlabeled_pool if int(node) not in used_nodes]

    if not eligible:
        global_unlabeled = torch.where(
            torch.bitwise_not(clean_data.train_mask)
            & torch.bitwise_not(clean_data.val_mask)
            & torch.bitwise_not(clean_data.test_mask)
            & getattr(clean_data, "unlabeled_mask", torch.ones(clean_data.num_nodes, dtype=torch.bool, device=clean_data.y.device))
            & (clean_data.clean_y != int(target_label))
        )[0].tolist()
        eligible = [int(node) for node in global_unlabeled if int(node) not in used_nodes]

    desired = max(1, int(np.floor(len(owned_train_nodes) * poison_rate)))
    desired = min(desired, len(eligible))
    if desired <= 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=desired, replace=False).tolist())


def _base_edge_weight(data: Data) -> torch.Tensor:
    if hasattr(data, "edge_weight") and getattr(data, "edge_weight") is not None:
        return data.edge_weight.clone()
    return torch.ones(data.edge_index.size(1), dtype=torch.float32, device=data.edge_index.device)


def inject_trigger_for_nodes(
    data: Data,
    victim_nodes: Sequence[int],
    shape_for_victim: Dict[int, List[Tuple[int, int]]],
    target_label: int,
    anchor_mode: str = "random",
    seed: int = 0,
    trigger_feature_mode: str = "prototype_mix",
    drop_existing_victim_edge: bool = False,
    anchor_connection_mode: str = "all",
    trigger_features_for_victim: Dict[int, torch.Tensor] | None = None,
    trigger_edge_weights_for_victim: Dict[int, List[float]] | None = None,
) -> Data:
    attacked = deepcopy(data)
    if not hasattr(attacked, "clean_y"):
        attacked.clean_y = attacked.y.clone()
    if not hasattr(attacked, "poisoned_target_mask"):
        attacked.poisoned_target_mask = torch.zeros(attacked.num_nodes, dtype=torch.bool, device=attacked.y.device)

    edge_pairs: List[Tuple[int, int]] = attacked.edge_index.t().tolist()
    edge_weights = _base_edge_weight(attacked).detach().cpu().tolist()

    prototype = build_target_feature_prototype(attacked, int(target_label)).to(attacked.x.device)
    degrees = torch.bincount(attacked.edge_index[0].cpu(), minlength=attacked.num_nodes)
    rng = np.random.default_rng(seed)

    new_features: List[torch.Tensor] = []
    new_labels: List[int] = []
    new_train_mask: List[bool] = []
    new_val_mask: List[bool] = []
    new_test_mask: List[bool] = []
    new_labeled_mask: List[bool] = []
    new_unlabeled_mask: List[bool] = []
    new_poison_mask: List[bool] = []

    current_num_nodes = attacked.num_nodes
    for offset, victim in enumerate(victim_nodes):
        internal_edges = shape_for_victim[int(victim)]
        victim_feature = attacked.x[int(victim)].detach().clone()

        custom_trigger_features = None
        if trigger_features_for_victim is not None and int(victim) in trigger_features_for_victim:
            custom_trigger_features = trigger_features_for_victim[int(victim)].to(attacked.x.device).float()
        if custom_trigger_features is not None:
            trigger_num_nodes = int(custom_trigger_features.size(0))
        else:
            trigger_num_nodes = 1 + max(max(src, dst) for src, dst in internal_edges) if internal_edges else 1

        if custom_trigger_features is not None:
            if custom_trigger_features.dim() != 2 or custom_trigger_features.size(0) != trigger_num_nodes:
                raise ValueError(
                    f"Trigger features for victim {victim} have shape {tuple(custom_trigger_features.shape)}, "
                    f"expected ({trigger_num_nodes}, {attacked.num_features})."
                )

        if custom_trigger_features is not None:
            trigger_features = custom_trigger_features
        elif trigger_feature_mode == "prototype_mix":
            noise = torch.randn(trigger_num_nodes, attacked.num_features, device=attacked.x.device) * 0.02
            trigger_features = (
                0.55 * prototype.unsqueeze(0).repeat(trigger_num_nodes, 1)
                + 0.35 * victim_feature.unsqueeze(0).repeat(trigger_num_nodes, 1)
                + 0.10 * noise
            )
        else:
            trigger_features = prototype.unsqueeze(0).repeat(trigger_num_nodes, 1)

        trigger_node_ids = list(range(current_num_nodes, current_num_nodes + trigger_num_nodes))
        for trigger_idx in range(trigger_num_nodes):
            new_features.append(trigger_features[trigger_idx : trigger_idx + 1])
            new_labels.append(-1)
            new_train_mask.append(False)
            new_val_mask.append(False)
            new_test_mask.append(False)
            new_labeled_mask.append(False)
            new_unlabeled_mask.append(False)
            new_poison_mask.append(False)

        if anchor_mode == "random":
            # "random injection position" is already realized when victim/attach nodes are randomly sampled.
            # Using the victim as the anchor matches the reference Renyi baseline more closely and avoids
            # weakening the trigger by randomly drifting to a neighbor node.
            anchor_node = int(victim)
        else:
            anchor_node = int(victim)

        if anchor_connection_mode == "single_root":
            anchor_targets = trigger_node_ids[:1]
        else:
            anchor_targets = trigger_node_ids
        for trigger_node in anchor_targets:
            edge_pairs.append((anchor_node, trigger_node))
            edge_pairs.append((trigger_node, anchor_node))
            edge_weights.extend([1.0, 1.0])

        weight_list = trigger_edge_weights_for_victim.get(int(victim), []) if trigger_edge_weights_for_victim else []
        for edge_idx, (src, dst) in enumerate(internal_edges):
            real_src = trigger_node_ids[src]
            real_dst = trigger_node_ids[dst]
            edge_weight = float(weight_list[edge_idx]) if edge_idx < len(weight_list) else 1.0
            edge_pairs.append((real_src, real_dst))
            edge_pairs.append((real_dst, real_src))
            edge_weights.extend([edge_weight, edge_weight])

        if drop_existing_victim_edge and degrees[int(victim)].item() > 0:
            outgoing = attacked.edge_index[1][attacked.edge_index[0] == int(victim)].detach().cpu().tolist()
            if outgoing:
                removed_neighbor = int(outgoing[(offset + seed) % len(outgoing)])
                try:
                    forward_idx = edge_pairs.index((int(victim), removed_neighbor))
                    backward_idx = edge_pairs.index((removed_neighbor, int(victim)))
                    for idx in sorted([forward_idx, backward_idx], reverse=True):
                        edge_pairs.pop(idx)
                        edge_weights.pop(idx)
                except ValueError:
                    pass

        attacked.y[int(victim)] = int(target_label)
        attacked.poisoned_target_mask[int(victim)] = True
        current_num_nodes += trigger_num_nodes

    if new_features:
        attacked.x = torch.cat([attacked.x] + new_features, dim=0)
        attacked.y = torch.cat(
            [attacked.y, torch.tensor(new_labels, dtype=attacked.y.dtype, device=attacked.y.device)],
            dim=0,
        )
        attacked.clean_y = torch.cat(
            [attacked.clean_y, torch.tensor(new_labels, dtype=attacked.clean_y.dtype, device=attacked.clean_y.device)],
            dim=0,
        )
        attacked.train_mask = torch.cat(
            [attacked.train_mask, torch.tensor(new_train_mask, dtype=torch.bool, device=attacked.train_mask.device)],
            dim=0,
        )
        attacked.val_mask = torch.cat(
            [attacked.val_mask, torch.tensor(new_val_mask, dtype=torch.bool, device=attacked.val_mask.device)],
            dim=0,
        )
        attacked.test_mask = torch.cat(
            [attacked.test_mask, torch.tensor(new_test_mask, dtype=torch.bool, device=attacked.test_mask.device)],
            dim=0,
        )
        if hasattr(attacked, "labeled_mask"):
            attacked.labeled_mask = torch.cat(
                [
                    attacked.labeled_mask,
                    torch.tensor(new_labeled_mask, dtype=torch.bool, device=attacked.labeled_mask.device),
                ],
                dim=0,
            )
        if hasattr(attacked, "unlabeled_mask"):
            attacked.unlabeled_mask = torch.cat(
                [
                    attacked.unlabeled_mask,
                    torch.tensor(new_unlabeled_mask, dtype=torch.bool, device=attacked.unlabeled_mask.device),
                ],
                dim=0,
            )
        for mask_name in ("clean_test_mask", "attack_test_mask"):
            if hasattr(attacked, mask_name):
                old_mask = getattr(attacked, mask_name)
                setattr(
                    attacked,
                    mask_name,
                    torch.cat(
                        [old_mask, torch.tensor(new_test_mask, dtype=torch.bool, device=old_mask.device)],
                        dim=0,
                    ),
                )
        attacked.poisoned_target_mask = torch.cat(
            [
                attacked.poisoned_target_mask,
                torch.tensor(new_poison_mask, dtype=torch.bool, device=attacked.poisoned_target_mask.device),
            ],
            dim=0,
        )

    attacked.edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=attacked.edge_index.device).t().contiguous()
    attacked.edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=attacked.edge_index.device)
    refresh_split_indices(attacked, target_label=target_label)
    return attacked


def _serialize_edge_map(edge_map: Dict[int, List[Tuple[int, int]]]) -> Dict[str, List[List[int]]]:
    return {str(key): [[int(src), int(dst)] for src, dst in value] for key, value in edge_map.items()}


def _build_static_trigger_edges_from_cfg(
    attack_cfg: Dict[str, object],
    seed: int,
) -> Tuple[str, List[Tuple[int, int]], Dict[str, object]]:
    attack_name = canonical_attack_name(str(attack_cfg.get("name", "multi_client_renyi_trigger")))
    trigger_num_nodes = int(attack_cfg["trigger_num_nodes"])
    trigger_tag = attack_tag_from_name(attack_name)

    if trigger_tag == "renyi":
        edges = randomize_trigger_edges(
            edges=build_renyi_trigger_edges(
                trigger_num_nodes=trigger_num_nodes,
                edge_prob=float(attack_cfg.get("renyi_edge_prob", 0.5)),
                seed=seed + 17,
            ),
            trigger_num_nodes=trigger_num_nodes,
            add_prob=float(attack_cfg.get("random_add_prob", 0.15)),
            drop_prob=float(attack_cfg.get("random_drop_prob", 0.15)),
            seed=seed + 23,
        )
        details = {
            "renyi_edge_prob": float(attack_cfg.get("renyi_edge_prob", 0.5)),
            "random_add_prob": float(attack_cfg.get("random_add_prob", 0.15)),
            "random_drop_prob": float(attack_cfg.get("random_drop_prob", 0.15)),
        }
    elif trigger_tag == "ba":
        edges = build_barabasi_albert_trigger_edges(
            trigger_num_nodes=trigger_num_nodes,
            degree=int(attack_cfg.get("ba_degree", attack_cfg.get("trigger_degree", 2))),
            seed=seed + 17,
        )
        details = {
            "ba_degree": int(attack_cfg.get("ba_degree", attack_cfg.get("trigger_degree", 2))),
        }
    elif trigger_tag == "ws":
        edges = build_watts_strogatz_trigger_edges(
            trigger_num_nodes=trigger_num_nodes,
            degree=int(attack_cfg.get("ws_degree", attack_cfg.get("trigger_degree", 2))),
            rewire_prob=float(attack_cfg.get("ws_rewire_prob", attack_cfg.get("renyi_edge_prob", 0.5))),
            seed=seed + 17,
        )
        details = {
            "ws_degree": int(attack_cfg.get("ws_degree", attack_cfg.get("trigger_degree", 2))),
            "ws_rewire_prob": float(attack_cfg.get("ws_rewire_prob", attack_cfg.get("renyi_edge_prob", 0.5))),
        }
    else:
        raise ValueError(f"Unsupported static trigger type: {attack_name}")

    return attack_name, edges, details


def _gta_generator_outputs(
    generator: GraphTrojanNet,
    graph: Data,
    victim_nodes: Sequence[int],
    trigger_num_nodes: int,
    edge_threshold: float,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[Tuple[int, int]]], Dict[int, List[float]]]:
    if not victim_nodes:
        return {}, {}, {}

    template_edges = _complete_trigger_edge_template(trigger_num_nodes)
    victim_tensor = torch.tensor(list(victim_nodes), dtype=torch.long, device=graph.x.device)
    with torch.set_grad_enabled(generator.training):
        trigger_features, edge_logits = generator(graph.x[victim_tensor].float())
        edge_probs = torch.sigmoid(edge_logits)

    feature_map: Dict[int, torch.Tensor] = {}
    edge_map: Dict[int, List[Tuple[int, int]]] = {}
    edge_weight_map: Dict[int, List[float]] = {}
    for row_idx, victim_node in enumerate(victim_nodes):
        feature_map[int(victim_node)] = trigger_features[row_idx]
        probs_row = edge_probs[row_idx] if edge_probs.numel() > 0 else torch.empty(0, device=graph.x.device)
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        for edge_idx, edge in enumerate(template_edges):
            if edge_idx < probs_row.numel():
                prob = float(probs_row[edge_idx].detach().cpu().item())
            else:
                prob = 1.0
            if prob >= float(edge_threshold):
                edges.append(edge)
                weights.append(prob)
        if not edges and template_edges:
            fallback_idx = int(probs_row.argmax().item()) if probs_row.numel() > 0 else 0
            edges = [template_edges[fallback_idx]]
            weights = [float(probs_row[fallback_idx].detach().cpu().item()) if probs_row.numel() > 0 else 1.0]
        edge_map[int(victim_node)] = edges
        edge_weight_map[int(victim_node)] = weights
    return feature_map, edge_map, edge_weight_map


def _fit_gta_generator(
    local_graph: Data,
    attach_nodes: Sequence[int],
    target_label: int,
    attack_cfg: Dict[str, object],
    seed: int,
) -> Tuple[GraphTrojanNet, Dict[int, torch.Tensor], Dict[int, List[Tuple[int, int]]], Dict[int, List[float]], float]:
    preferred_device = str(attack_cfg.get("gta_preconstruct_device", "cuda")).strip().lower()
    if preferred_device == "cuda" and torch.cuda.is_available():
        gta_device = torch.device("cuda")
    else:
        gta_device = local_graph.x.device

    if not attach_nodes:
        trigger_num_nodes = int(attack_cfg["trigger_num_nodes"])
        generator = GraphTrojanNet(
            input_dim=int(local_graph.num_features),
            trigger_num_nodes=trigger_num_nodes,
            hidden_dim=int(attack_cfg.get("gta_hidden_dim", max(32, min(128, int(local_graph.num_features))))),
            dropout=float(attack_cfg.get("gta_dropout", 0.0)),
        ).to(gta_device)
        generator.eval()
        return generator, {}, {}, {}, float("nan")

    device = gta_device
    trigger_num_nodes = int(attack_cfg["trigger_num_nodes"])
    generator = GraphTrojanNet(
        input_dim=int(local_graph.num_features),
        trigger_num_nodes=trigger_num_nodes,
        hidden_dim=int(attack_cfg.get("gta_hidden_dim", max(32, min(128, int(local_graph.num_features))))),
        dropout=float(attack_cfg.get("gta_dropout", 0.0)),
    ).to(device)
    shadow_model = NodeGCN(
        input_dim=int(local_graph.num_features),
        hidden_dim=int(attack_cfg.get("gta_shadow_hidden_dim", 64)),
        output_dim=int(local_graph.clean_y.max().item()) + 1,
        num_layers=int(attack_cfg.get("gta_shadow_num_layers", 3)),
        dropout=float(attack_cfg.get("gta_shadow_dropout", 0.0)),
        layer_norm_first=True,
        use_ln=True,
    ).to(device)

    cpu_train_graph = deepcopy(local_graph)
    train_graph = deepcopy(local_graph)
    train_graph.edge_index = getattr(local_graph, "train_edge_index", local_graph.edge_index).clone().to(device)
    train_graph.edge_weight = getattr(local_graph, "train_edge_weight", getattr(local_graph, "edge_weight", None))
    if train_graph.edge_weight is None:
        train_graph.edge_weight = torch.ones(train_graph.edge_index.size(1), dtype=torch.float32, device=device)
    else:
        train_graph.edge_weight = train_graph.edge_weight.clone().to(device)
    train_graph.x = train_graph.x.clone().to(device)
    train_graph.y = train_graph.y.clone().to(device)
    train_graph.clean_y = train_graph.clean_y.clone().to(device)
    train_graph.train_mask = train_graph.train_mask.clone().to(device)
    train_graph.val_mask = train_graph.val_mask.clone().to(device)
    train_graph.test_mask = train_graph.test_mask.clone().to(device)
    if hasattr(train_graph, "labeled_mask"):
        train_graph.labeled_mask = train_graph.labeled_mask.clone().to(device)
    if hasattr(train_graph, "unlabeled_mask"):
        train_graph.unlabeled_mask = train_graph.unlabeled_mask.clone().to(device)

    prototype = build_target_feature_prototype(train_graph, int(target_label)).to(device)
    attach_tensor = torch.tensor(list(attach_nodes), dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(
        list(generator.parameters()) + list(shadow_model.parameters()),
        lr=float(attack_cfg.get("gta_lr", 0.01)),
        weight_decay=float(attack_cfg.get("gta_weight_decay", 5e-4)),
    )

    edge_threshold = float(attack_cfg.get("gta_edge_threshold", 0.5))
    feature_reg_weight = float(attack_cfg.get("gta_feature_reg_weight", 0.1))
    edge_reg_weight = float(attack_cfg.get("gta_edge_reg_weight", 0.05))
    last_loss = float("nan")

    for _ in range(int(attack_cfg.get("gta_trojan_epochs", 60))):
        generator.train()
        shadow_model.train()
        optimizer.zero_grad()

        feature_map, edge_map, edge_weight_map = _gta_generator_outputs(
            generator=generator,
            graph=train_graph,
            victim_nodes=attach_nodes,
            trigger_num_nodes=trigger_num_nodes,
            edge_threshold=edge_threshold,
        )
        attacked_graph = inject_trigger_for_nodes(
            data=cpu_train_graph,
            victim_nodes=attach_nodes,
            shape_for_victim=edge_map,
            target_label=int(target_label),
            anchor_mode=str(attack_cfg.get("anchor_mode", "random")),
            seed=seed,
            trigger_feature_mode="gta_generator",
            drop_existing_victim_edge=bool(attack_cfg.get("drop_existing_victim_edge", False)),
            anchor_connection_mode="single_root",
            trigger_features_for_victim=feature_map,
            trigger_edge_weights_for_victim=edge_weight_map,
        )
        attacked_graph = attacked_graph.to(device)
        poisoned_train_mask = attacked_graph.train_mask.clone()
        poisoned_train_mask[attach_tensor] = True
        target_labels = attacked_graph.clean_y.clone()
        target_labels[attach_tensor] = int(target_label)

        logits = shadow_model(attacked_graph.x, attacked_graph.edge_index, attacked_graph.edge_weight)
        cls_loss = F.cross_entropy(logits[poisoned_train_mask], target_labels[poisoned_train_mask])

        stacked_features = torch.stack([feature_map[int(node)] for node in attach_nodes], dim=0)
        feat_loss = F.mse_loss(
            stacked_features,
            prototype.view(1, 1, -1).expand(stacked_features.size(0), stacked_features.size(1), prototype.numel()),
        )
        if edge_weight_map:
            edge_values = torch.tensor(
                [weight for node in attach_nodes for weight in edge_weight_map[int(node)]],
                dtype=torch.float32,
                device=device,
            )
            edge_loss = (1.0 - edge_values).mean() if edge_values.numel() > 0 else torch.tensor(0.0, device=device)
        else:
            edge_loss = torch.tensor(0.0, device=device)

        loss = cls_loss + feature_reg_weight * feat_loss + edge_reg_weight * edge_loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

    generator.eval()
    with torch.no_grad():
        feature_map, edge_map, edge_weight_map = _gta_generator_outputs(
            generator=generator,
            graph=train_graph,
            victim_nodes=attach_nodes,
            trigger_num_nodes=trigger_num_nodes,
            edge_threshold=edge_threshold,
        )
    return generator.cpu(), feature_map, edge_map, edge_weight_map, last_loss


def build_multi_client_renyi_attack(
    clean_data: Data,
    client_node_indices: Sequence[Sequence[int]],
    attack_cfg: Dict[str, object],
    seed: int,
) -> Tuple[List[Data], Dict[str, object], Dict[str, object]]:
    target_label = choose_attack_target_label(clean_data, int(attack_cfg["target_label"]))
    malicious_client_ids = select_malicious_clients(
        client_node_indices=client_node_indices,
        malicious_fraction=float(attack_cfg["malicious_client_fraction"]),
        seed=seed,
    )
    global_template_edges = randomize_trigger_edges(
        edges=build_renyi_trigger_edges(
            trigger_num_nodes=int(attack_cfg["trigger_num_nodes"]),
            edge_prob=float(attack_cfg["renyi_edge_prob"]),
            seed=seed + 17,
        ),
        trigger_num_nodes=int(attack_cfg["trigger_num_nodes"]),
        add_prob=float(attack_cfg.get("random_add_prob", 0.15)),
        drop_prob=float(attack_cfg.get("random_drop_prob", 0.15)),
        seed=seed + 23,
    )

    local_graphs = [deepcopy(clean_data) for _ in client_node_indices]
    for client_id, node_indices in enumerate(client_node_indices):
        local_graphs[client_id].owned_train_mask = torch.zeros(local_graphs[client_id].num_nodes, dtype=torch.bool)
        local_graphs[client_id].owned_train_mask[torch.tensor(node_indices, dtype=torch.long)] = True
        local_graphs[client_id].poisoned_target_mask = torch.zeros(local_graphs[client_id].num_nodes, dtype=torch.bool)

    client_metadata: List[Dict[str, object]] = []
    global_poisoned_nodes: List[int] = []
    trigger_edges_by_client: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
    used_poison_nodes: Set[int] = set()

    for order, client_id in enumerate(malicious_client_ids):
        selected_nodes = select_poisoned_nodes_for_client(
            clean_data=clean_data,
            owned_train_nodes=client_node_indices[client_id],
            target_label=target_label,
            poison_rate=float(attack_cfg["poison_rate"]),
            seed=seed + 97 * (order + 1),
            used_nodes=used_poison_nodes,
        )
        shape_for_victim: Dict[int, List[Tuple[int, int]]] = {}
        for victim_node in selected_nodes:
            shape_for_victim[int(victim_node)] = list(global_template_edges)

        attacked_graph = inject_trigger_for_nodes(
            data=clean_data,
            victim_nodes=selected_nodes,
            shape_for_victim=shape_for_victim,
            target_label=int(target_label),
            anchor_mode=str(attack_cfg.get("anchor_mode", "random")),
            seed=seed + 3000 * (order + 1),
            trigger_feature_mode=str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
            drop_existing_victim_edge=bool(attack_cfg.get("drop_existing_victim_edge", False)),
        )
        attacked_graph.owned_train_mask = torch.zeros(attacked_graph.num_nodes, dtype=torch.bool)
        attacked_graph.owned_train_mask[torch.tensor(client_node_indices[client_id], dtype=torch.long)] = True
        if selected_nodes:
            attacked_graph.owned_train_mask[torch.tensor(selected_nodes, dtype=torch.long)] = True
        local_graphs[client_id] = attacked_graph

        global_poisoned_nodes.extend(int(node) for node in selected_nodes)
        used_poison_nodes.update(int(node) for node in selected_nodes)
        trigger_edges_by_client[int(client_id)] = shape_for_victim
        client_metadata.append(
            {
                "client_id": int(client_id),
                "num_owned_train_nodes": int(len(client_node_indices[client_id])),
                "num_poisoned_nodes": int(len(selected_nodes)),
                "poisoned_nodes": [int(node) for node in selected_nodes],
                "shape_for_victim": _serialize_edge_map(shape_for_victim),
            }
        )

    attack_metadata = {
        "attack_name": "multi_client_renyi_trigger",
        "target_label": int(target_label),
        "malicious_client_fraction": float(attack_cfg["malicious_client_fraction"]),
        "malicious_client_ids": [int(client_id) for client_id in malicious_client_ids],
        "num_malicious_clients": int(len(malicious_client_ids)),
        "poison_rate": float(attack_cfg["poison_rate"]),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "renyi_edge_prob": float(attack_cfg["renyi_edge_prob"]),
        "random_add_prob": float(attack_cfg.get("random_add_prob", 0.15)),
        "random_drop_prob": float(attack_cfg.get("random_drop_prob", 0.15)),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "trigger_feature_mode": str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "global_trigger_edges": [[int(src), int(dst)] for src, dst in global_template_edges],
        "clients": client_metadata,
        "all_poisoned_nodes": sorted(set(global_poisoned_nodes)),
    }
    runtime = {
        "attack_name": "multi_client_renyi_trigger",
        "target_label": int(target_label),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "renyi_edge_prob": float(attack_cfg["renyi_edge_prob"]),
        "random_add_prob": float(attack_cfg.get("random_add_prob", 0.15)),
        "random_drop_prob": float(attack_cfg.get("random_drop_prob", 0.15)),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "trigger_feature_mode": str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "global_trigger_edges": [tuple(map(int, edge)) for edge in global_template_edges],
        "client_shape_map": trigger_edges_by_client,
    }
    return local_graphs, attack_metadata, runtime


def build_local_client_renyi_attack(
    local_graphs: Sequence[Data],
    attack_cfg: Dict[str, object],
    seed: int,
) -> Tuple[List[Data], Dict[str, object], Dict[str, object]]:
    target_label = choose_attack_target_label_from_graphs(local_graphs, int(attack_cfg["target_label"]))
    client_train_nodes = [torch.where(graph.train_mask)[0].tolist() for graph in local_graphs]
    malicious_client_ids = select_malicious_clients(
        client_node_indices=client_train_nodes,
        malicious_fraction=float(attack_cfg["malicious_client_fraction"]),
        seed=seed,
    )
    global_template_edges = randomize_trigger_edges(
        edges=build_renyi_trigger_edges(
            trigger_num_nodes=int(attack_cfg["trigger_num_nodes"]),
            edge_prob=float(attack_cfg["renyi_edge_prob"]),
            seed=seed + 17,
        ),
        trigger_num_nodes=int(attack_cfg["trigger_num_nodes"]),
        add_prob=float(attack_cfg.get("random_add_prob", 0.15)),
        drop_prob=float(attack_cfg.get("random_drop_prob", 0.15)),
        seed=seed + 23,
    )

    attacked_local_graphs = [deepcopy(graph) for graph in local_graphs]
    client_metadata: List[Dict[str, object]] = []
    all_poisoned_nodes: List[int] = []

    for order, client_id in enumerate(malicious_client_ids):
        local_graph = deepcopy(local_graphs[client_id])
        base_train_graph = deepcopy(local_graph)
        base_train_graph.edge_index = local_graph.train_edge_index.clone()
        base_train_graph.edge_weight = local_graph.train_edge_weight.clone()
        attach_nodes = _select_attach_nodes_local(
            local_graph=local_graph,
            target_label=target_label,
            poison_rate=float(attack_cfg["poison_rate"]),
            seed=seed + 97 * (order + 1),
        )
        shape_for_victim = {int(node): list(global_template_edges) for node in attach_nodes}
        attacked_graph = inject_trigger_for_nodes(
            data=base_train_graph,
            victim_nodes=attach_nodes,
            shape_for_victim=shape_for_victim,
            target_label=int(target_label),
            anchor_mode=str(attack_cfg.get("anchor_mode", "random")),
            seed=seed + 3000 * (order + 1),
            trigger_feature_mode=str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
            drop_existing_victim_edge=bool(attack_cfg.get("drop_existing_victim_edge", False)),
        )
        attacked_graph.client_id = int(client_id)
        extra_nodes = attacked_graph.num_nodes - local_graph.num_nodes
        if extra_nodes > 0:
            attacked_graph.index_orig = torch.cat(
                [local_graph.index_orig.clone(), torch.full((extra_nodes,), -1, dtype=local_graph.index_orig.dtype)],
                dim=0,
            )
        else:
            attacked_graph.index_orig = local_graph.index_orig.clone()
        attacked_graph.train_edge_index = attacked_graph.edge_index.clone()
        attacked_graph.train_edge_weight = attacked_graph.edge_weight.clone()
        false_pad = torch.zeros(extra_nodes, dtype=torch.bool)
        attacked_graph.train_mask = torch.cat([local_graph.train_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.val_mask = torch.cat([local_graph.val_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.test_mask = torch.cat([local_graph.test_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.labeled_mask = torch.cat([local_graph.labeled_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.unlabeled_mask = torch.cat([local_graph.unlabeled_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.clean_test_mask = attacked_graph.test_mask.clone()
        attacked_graph.attack_test_mask = attacked_graph.test_mask.clone()
        attacked_graph.owned_train_mask = attacked_graph.train_mask.clone()
        if attach_nodes:
            attacked_graph.owned_train_mask[torch.tensor(attach_nodes, dtype=torch.long)] = True
        refresh_split_indices(attacked_graph, target_label=target_label)
        attacked_local_graphs[client_id] = attacked_graph

        all_poisoned_nodes.extend(int(node) for node in attach_nodes)
        client_metadata.append(
            {
                "client_id": int(client_id),
                "num_local_nodes": int(local_graph.num_nodes),
                "num_local_train_nodes": int(local_graph.train_mask.sum().item()),
                "num_poisoned_nodes": int(len(attach_nodes)),
                "poisoned_nodes": [int(node) for node in attach_nodes],
                "shape_for_victim": _serialize_edge_map(shape_for_victim),
            }
        )

    attack_metadata = {
        "attack_name": "multi_client_renyi_trigger",
        "target_label": int(target_label),
        "malicious_client_fraction": float(attack_cfg["malicious_client_fraction"]),
        "malicious_client_ids": [int(client_id) for client_id in malicious_client_ids],
        "num_malicious_clients": int(len(malicious_client_ids)),
        "poison_rate": float(attack_cfg["poison_rate"]),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "renyi_edge_prob": float(attack_cfg["renyi_edge_prob"]),
        "random_add_prob": float(attack_cfg.get("random_add_prob", 0.15)),
        "random_drop_prob": float(attack_cfg.get("random_drop_prob", 0.15)),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "trigger_feature_mode": str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "global_trigger_edges": [[int(src), int(dst)] for src, dst in global_template_edges],
        "clients": client_metadata,
        "all_poisoned_nodes": sorted(set(all_poisoned_nodes)),
    }
    runtime = {
        "attack_name": "multi_client_renyi_trigger",
        "target_label": int(target_label),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "renyi_edge_prob": float(attack_cfg["renyi_edge_prob"]),
        "random_add_prob": float(attack_cfg.get("random_add_prob", 0.15)),
        "random_drop_prob": float(attack_cfg.get("random_drop_prob", 0.15)),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "trigger_feature_mode": str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "global_trigger_edges": [tuple(map(int, edge)) for edge in global_template_edges],
    }
    return attacked_local_graphs, attack_metadata, runtime


def build_local_client_static_trigger_attack(
    local_graphs: Sequence[Data],
    attack_cfg: Dict[str, object],
    seed: int,
) -> Tuple[List[Data], Dict[str, object], Dict[str, object]]:
    attack_name, global_template_edges, trigger_details = _build_static_trigger_edges_from_cfg(attack_cfg=attack_cfg, seed=seed)
    target_label = choose_attack_target_label_from_graphs(local_graphs, int(attack_cfg["target_label"]))
    client_train_nodes = [torch.where(graph.train_mask)[0].tolist() for graph in local_graphs]
    malicious_client_ids = select_malicious_clients(
        client_node_indices=client_train_nodes,
        malicious_fraction=float(attack_cfg["malicious_client_fraction"]),
        seed=seed,
    )

    attacked_local_graphs = [deepcopy(graph) for graph in local_graphs]
    client_metadata: List[Dict[str, object]] = []
    all_poisoned_nodes: List[int] = []
    anchor_connection_mode = str(attack_cfg.get("anchor_connection_mode", "single_root"))

    for order, client_id in enumerate(malicious_client_ids):
        local_graph = deepcopy(local_graphs[client_id])
        base_train_graph = deepcopy(local_graph)
        base_train_graph.edge_index = local_graph.train_edge_index.clone()
        base_train_graph.edge_weight = local_graph.train_edge_weight.clone()
        attach_nodes = _select_attach_nodes_local(
            local_graph=local_graph,
            target_label=target_label,
            poison_rate=float(attack_cfg["poison_rate"]),
            seed=seed + 97 * (order + 1),
        )
        shape_for_victim = {int(node): list(global_template_edges) for node in attach_nodes}
        attacked_graph = inject_trigger_for_nodes(
            data=base_train_graph,
            victim_nodes=attach_nodes,
            shape_for_victim=shape_for_victim,
            target_label=int(target_label),
            anchor_mode=str(attack_cfg.get("anchor_mode", "random")),
            seed=seed + 3000 * (order + 1),
            trigger_feature_mode=str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
            drop_existing_victim_edge=bool(attack_cfg.get("drop_existing_victim_edge", False)),
            anchor_connection_mode=anchor_connection_mode,
        )
        attacked_graph.client_id = int(client_id)
        extra_nodes = attacked_graph.num_nodes - local_graph.num_nodes
        if extra_nodes > 0:
            attacked_graph.index_orig = torch.cat(
                [local_graph.index_orig.clone(), torch.full((extra_nodes,), -1, dtype=local_graph.index_orig.dtype)],
                dim=0,
            )
        else:
            attacked_graph.index_orig = local_graph.index_orig.clone()
        attacked_graph.train_edge_index = attacked_graph.edge_index.clone()
        attacked_graph.train_edge_weight = attacked_graph.edge_weight.clone()
        false_pad = torch.zeros(extra_nodes, dtype=torch.bool)
        attacked_graph.train_mask = torch.cat([local_graph.train_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.val_mask = torch.cat([local_graph.val_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.test_mask = torch.cat([local_graph.test_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.labeled_mask = torch.cat([local_graph.labeled_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.unlabeled_mask = torch.cat([local_graph.unlabeled_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.clean_test_mask = attacked_graph.test_mask.clone()
        attacked_graph.attack_test_mask = attacked_graph.test_mask.clone()
        attacked_graph.owned_train_mask = attacked_graph.train_mask.clone()
        if attach_nodes:
            attacked_graph.owned_train_mask[torch.tensor(attach_nodes, dtype=torch.long)] = True
        refresh_split_indices(attacked_graph, target_label=target_label)
        attacked_local_graphs[client_id] = attacked_graph

        all_poisoned_nodes.extend(int(node) for node in attach_nodes)
        client_metadata.append(
            {
                "client_id": int(client_id),
                "num_local_nodes": int(local_graph.num_nodes),
                "num_local_train_nodes": int(local_graph.train_mask.sum().item()),
                "num_poisoned_nodes": int(len(attach_nodes)),
                "poisoned_nodes": [int(node) for node in attach_nodes],
                "shape_for_victim": _serialize_edge_map(shape_for_victim),
            }
        )

    attack_metadata = {
        "attack_name": attack_name,
        "target_label": int(target_label),
        "malicious_client_fraction": float(attack_cfg["malicious_client_fraction"]),
        "malicious_client_ids": [int(client_id) for client_id in malicious_client_ids],
        "num_malicious_clients": int(len(malicious_client_ids)),
        "poison_rate": float(attack_cfg["poison_rate"]),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "anchor_connection_mode": anchor_connection_mode,
        "trigger_feature_mode": str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "global_trigger_edges": [[int(src), int(dst)] for src, dst in global_template_edges],
        "clients": client_metadata,
        "all_poisoned_nodes": sorted(set(all_poisoned_nodes)),
        **trigger_details,
    }
    runtime = {
        "attack_name": attack_name,
        "target_label": int(target_label),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "anchor_connection_mode": anchor_connection_mode,
        "trigger_feature_mode": str(attack_cfg.get("trigger_feature_mode", "prototype_mix")),
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "global_trigger_edges": [tuple(map(int, edge)) for edge in global_template_edges],
        **trigger_details,
    }
    return attacked_local_graphs, attack_metadata, runtime


def build_local_client_gta_attack(
    local_graphs: Sequence[Data],
    attack_cfg: Dict[str, object],
    seed: int,
) -> Tuple[List[Data], Dict[str, object], Dict[str, object]]:
    attack_name = canonical_attack_name(str(attack_cfg.get("name", "multi_client_gta_trigger")))
    target_label = choose_attack_target_label_from_graphs(local_graphs, int(attack_cfg["target_label"]))
    client_train_nodes = [torch.where(graph.train_mask)[0].tolist() for graph in local_graphs]
    malicious_client_ids = select_malicious_clients(
        client_node_indices=client_train_nodes,
        malicious_fraction=float(attack_cfg["malicious_client_fraction"]),
        seed=seed,
    )

    attacked_local_graphs = [deepcopy(graph) for graph in local_graphs]
    client_metadata: List[Dict[str, object]] = []
    all_poisoned_nodes: List[int] = []
    runtime_generators: Dict[int, Dict[str, object]] = {}

    for order, client_id in enumerate(malicious_client_ids):
        local_graph = deepcopy(local_graphs[client_id])
        base_train_graph = deepcopy(local_graph)
        base_train_graph.edge_index = local_graph.train_edge_index.clone()
        base_train_graph.edge_weight = local_graph.train_edge_weight.clone()
        attach_nodes = _select_attach_nodes_local(
            local_graph=local_graph,
            target_label=target_label,
            poison_rate=float(attack_cfg["poison_rate"]),
            seed=seed + 97 * (order + 1),
        )
        generator, feature_map, edge_map, edge_weight_map, gta_loss = _fit_gta_generator(
            local_graph=base_train_graph,
            attach_nodes=attach_nodes,
            target_label=int(target_label),
            attack_cfg=attack_cfg,
            seed=seed + 3000 * (order + 1),
        )
        attacked_graph = inject_trigger_for_nodes(
            data=base_train_graph,
            victim_nodes=attach_nodes,
            shape_for_victim=edge_map,
            target_label=int(target_label),
            anchor_mode=str(attack_cfg.get("anchor_mode", "random")),
            seed=seed + 3000 * (order + 1),
            trigger_feature_mode="gta_generator",
            drop_existing_victim_edge=bool(attack_cfg.get("drop_existing_victim_edge", False)),
            anchor_connection_mode="single_root",
            trigger_features_for_victim=feature_map,
            trigger_edge_weights_for_victim=edge_weight_map,
        )
        attacked_graph.client_id = int(client_id)
        extra_nodes = attacked_graph.num_nodes - local_graph.num_nodes
        if extra_nodes > 0:
            attacked_graph.index_orig = torch.cat(
                [local_graph.index_orig.clone(), torch.full((extra_nodes,), -1, dtype=local_graph.index_orig.dtype)],
                dim=0,
            )
        else:
            attacked_graph.index_orig = local_graph.index_orig.clone()
        attacked_graph.train_edge_index = attacked_graph.edge_index.clone()
        attacked_graph.train_edge_weight = attacked_graph.edge_weight.clone()
        false_pad = torch.zeros(extra_nodes, dtype=torch.bool)
        attacked_graph.train_mask = torch.cat([local_graph.train_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.val_mask = torch.cat([local_graph.val_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.test_mask = torch.cat([local_graph.test_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.labeled_mask = torch.cat([local_graph.labeled_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.unlabeled_mask = torch.cat([local_graph.unlabeled_mask.clone(), false_pad.clone()], dim=0)
        attacked_graph.clean_test_mask = attacked_graph.test_mask.clone()
        attacked_graph.attack_test_mask = attacked_graph.test_mask.clone()
        attacked_graph.owned_train_mask = attacked_graph.train_mask.clone()
        if attach_nodes:
            attacked_graph.owned_train_mask[torch.tensor(attach_nodes, dtype=torch.long)] = True
        refresh_split_indices(attacked_graph, target_label=target_label)
        attacked_local_graphs[client_id] = attacked_graph

        runtime_generators[int(client_id)] = {
            "state_dict": {key: value.detach().cpu() for key, value in generator.state_dict().items()},
            "input_dim": int(local_graph.num_features),
            "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
            "hidden_dim": int(attack_cfg.get("gta_hidden_dim", max(32, min(128, int(local_graph.num_features))))),
            "dropout": float(attack_cfg.get("gta_dropout", 0.0)),
        }
        all_poisoned_nodes.extend(int(node) for node in attach_nodes)
        client_metadata.append(
            {
                "client_id": int(client_id),
                "num_local_nodes": int(local_graph.num_nodes),
                "num_local_train_nodes": int(local_graph.train_mask.sum().item()),
                "num_poisoned_nodes": int(len(attach_nodes)),
                "poisoned_nodes": [int(node) for node in attach_nodes],
                "shape_for_victim": _serialize_edge_map(edge_map),
                "gta_training_loss": float(gta_loss),
            }
        )

    attack_metadata = {
        "attack_name": attack_name,
        "target_label": int(target_label),
        "malicious_client_fraction": float(attack_cfg["malicious_client_fraction"]),
        "malicious_client_ids": [int(client_id) for client_id in malicious_client_ids],
        "num_malicious_clients": int(len(malicious_client_ids)),
        "poison_rate": float(attack_cfg["poison_rate"]),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "anchor_connection_mode": "single_root",
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "gta_edge_threshold": float(attack_cfg.get("gta_edge_threshold", 0.5)),
        "gta_trojan_epochs": int(attack_cfg.get("gta_trojan_epochs", 60)),
        "clients": client_metadata,
        "all_poisoned_nodes": sorted(set(all_poisoned_nodes)),
    }
    runtime = {
        "attack_name": attack_name,
        "target_label": int(target_label),
        "trigger_num_nodes": int(attack_cfg["trigger_num_nodes"]),
        "anchor_mode": str(attack_cfg.get("anchor_mode", "random")),
        "anchor_connection_mode": "single_root",
        "drop_existing_victim_edge": bool(attack_cfg.get("drop_existing_victim_edge", False)),
        "gta_edge_threshold": float(attack_cfg.get("gta_edge_threshold", 0.5)),
        "gta_client_generators": runtime_generators,
        "default_eval_client_id": int(malicious_client_ids[0]) if malicious_client_ids else -1,
    }
    return attacked_local_graphs, attack_metadata, runtime


def build_local_client_attack(
    local_graphs: Sequence[Data],
    attack_cfg: Dict[str, object],
    seed: int,
) -> Tuple[List[Data], Dict[str, object], Dict[str, object]]:
    attack_name = canonical_attack_name(str(attack_cfg.get("name", "multi_client_renyi_trigger")))
    attack_cfg = dict(attack_cfg)
    attack_cfg["name"] = attack_name
    if attack_name == "multi_client_renyi_trigger":
        return build_local_client_renyi_attack(local_graphs=local_graphs, attack_cfg=attack_cfg, seed=seed)
    if attack_name in {"multi_client_ba_trigger", "multi_client_ws_trigger"}:
        return build_local_client_static_trigger_attack(local_graphs=local_graphs, attack_cfg=attack_cfg, seed=seed)
    if attack_name == "multi_client_gta_trigger":
        return build_local_client_gta_attack(local_graphs=local_graphs, attack_cfg=attack_cfg, seed=seed)
    raise ValueError(f"Unsupported attack: {attack_name}")


@torch.no_grad()
def build_eval_attack_graph_for_nodes(
    clean_data: Data,
    victim_nodes: Sequence[int],
    attack_runtime: Dict[str, object],
    seed: int,
    source_client_id: int | None = None,
) -> Data:
    attack_name = canonical_attack_name(str(attack_runtime.get("attack_name", "multi_client_renyi_trigger")))
    target_label = int(attack_runtime["target_label"])
    victim_nodes = [int(node) for node in victim_nodes]
    if attack_name == "multi_client_gta_trigger":
        runtime_generators = attack_runtime.get("gta_client_generators", {})
        if source_client_id is None:
            source_client_id = int(attack_runtime.get("default_eval_client_id", -1))
        generator_payload = runtime_generators.get(int(source_client_id))
        if generator_payload is None:
            raise ValueError(f"Missing GTA generator runtime for source client {source_client_id}.")
        generator = GraphTrojanNet(
            input_dim=int(generator_payload["input_dim"]),
            trigger_num_nodes=int(generator_payload["trigger_num_nodes"]),
            hidden_dim=int(generator_payload["hidden_dim"]),
            dropout=float(generator_payload.get("dropout", 0.0)),
        ).to(clean_data.x.device)
        generator.load_state_dict(generator_payload["state_dict"])
        generator.eval()
        feature_map, edge_map, edge_weight_map = _gta_generator_outputs(
            generator=generator,
            graph=clean_data,
            victim_nodes=victim_nodes,
            trigger_num_nodes=int(attack_runtime["trigger_num_nodes"]),
            edge_threshold=float(attack_runtime.get("gta_edge_threshold", 0.5)),
        )
        attacked = inject_trigger_for_nodes(
            data=clean_data,
            victim_nodes=victim_nodes,
            shape_for_victim=edge_map,
            target_label=target_label,
            anchor_mode=str(attack_runtime.get("anchor_mode", "random")),
            seed=seed,
            trigger_feature_mode="gta_generator",
            drop_existing_victim_edge=bool(attack_runtime.get("drop_existing_victim_edge", False)),
            anchor_connection_mode=str(attack_runtime.get("anchor_connection_mode", "single_root")),
            trigger_features_for_victim=feature_map,
            trigger_edge_weights_for_victim=edge_weight_map,
        )
    else:
        trigger_edges = [
            (int(src), int(dst))
            for src, dst in attack_runtime.get("global_trigger_edges", [])
        ]
        if not trigger_edges:
            trigger_num_nodes = int(attack_runtime["trigger_num_nodes"])
            if attack_name == "multi_client_ba_trigger":
                trigger_edges = build_barabasi_albert_trigger_edges(
                    trigger_num_nodes=trigger_num_nodes,
                    degree=int(attack_runtime.get("ba_degree", 2)),
                    seed=seed + 17,
                )
            elif attack_name == "multi_client_ws_trigger":
                trigger_edges = build_watts_strogatz_trigger_edges(
                    trigger_num_nodes=trigger_num_nodes,
                    degree=int(attack_runtime.get("ws_degree", 2)),
                    rewire_prob=float(attack_runtime.get("ws_rewire_prob", 0.5)),
                    seed=seed + 17,
                )
            else:
                base_edges = build_renyi_trigger_edges(
                    trigger_num_nodes=trigger_num_nodes,
                    edge_prob=float(attack_runtime.get("renyi_edge_prob", 0.5)),
                    seed=seed + 17,
                )
                trigger_edges = randomize_trigger_edges(
                    edges=base_edges,
                    trigger_num_nodes=trigger_num_nodes,
                    add_prob=float(attack_runtime.get("random_add_prob", 0.15)),
                    drop_prob=float(attack_runtime.get("random_drop_prob", 0.15)),
                    seed=seed + 23,
                )
        attacked = inject_trigger_for_nodes(
            data=clean_data,
            victim_nodes=victim_nodes,
            shape_for_victim={int(victim_node): trigger_edges for victim_node in victim_nodes},
            target_label=target_label,
            anchor_mode=str(attack_runtime.get("anchor_mode", "random")),
            seed=seed,
            trigger_feature_mode=str(attack_runtime.get("trigger_feature_mode", "prototype_mix")),
            drop_existing_victim_edge=bool(attack_runtime.get("drop_existing_victim_edge", False)),
            anchor_connection_mode=str(attack_runtime.get("anchor_connection_mode", "all")),
        )
    attacked.triggered_eval_mask = torch.zeros(attacked.num_nodes, dtype=torch.bool)
    if victim_nodes:
        attacked.triggered_eval_mask[torch.tensor(victim_nodes, dtype=torch.long, device=attacked.triggered_eval_mask.device)] = True
    return attacked


@torch.no_grad()
def build_single_eval_attack_graph(
    clean_data: Data,
    victim_node: int,
    attack_runtime: Dict[str, object],
    seed: int,
    source_client_id: int | None = None,
) -> Data:
    return build_eval_attack_graph_for_nodes(
        clean_data=clean_data,
        victim_nodes=[int(victim_node)],
        attack_runtime=attack_runtime,
        seed=seed + 13 * int(victim_node),
        source_client_id=source_client_id,
    )


def rebuild_local_graphs_from_metadata(
    clean_data: Data,
    partition_metadata: Dict[str, object],
    attack_metadata: Dict[str, object] | None,
) -> List[Data]:
    client_node_indices = [client["train_node_indices"] for client in partition_metadata["clients"]]
    local_graphs = [deepcopy(clean_data) for _ in client_node_indices]
    for client_id, node_indices in enumerate(client_node_indices):
        local_graphs[client_id].owned_train_mask = torch.zeros(local_graphs[client_id].num_nodes, dtype=torch.bool)
        local_graphs[client_id].owned_train_mask[torch.tensor(node_indices, dtype=torch.long)] = True
        local_graphs[client_id].poisoned_target_mask = torch.zeros(local_graphs[client_id].num_nodes, dtype=torch.bool)

    if not attack_metadata:
        return local_graphs

    target_label = int(attack_metadata["target_label"])
    for client_info in attack_metadata["clients"]:
        client_id = int(client_info["client_id"])
        poisoned_nodes = [int(node) for node in client_info["poisoned_nodes"]]
        shape_for_victim = {
            int(node): [tuple(edge) for edge in client_info["shape_for_victim"][str(node)]]
            for node in poisoned_nodes
        }
        attacked_graph = inject_trigger_for_nodes(
            data=clean_data,
            victim_nodes=poisoned_nodes,
            shape_for_victim=shape_for_victim,
            target_label=target_label,
            anchor_mode=str(attack_metadata.get("anchor_mode", "random")),
            seed=client_id,
            trigger_feature_mode=str(attack_metadata.get("trigger_feature_mode", "prototype_mix")),
        )
        attacked_graph.owned_train_mask = torch.zeros(attacked_graph.num_nodes, dtype=torch.bool)
        attacked_graph.owned_train_mask[torch.tensor(client_node_indices[client_id], dtype=torch.long)] = True
        local_graphs[client_id] = attacked_graph
    return local_graphs
