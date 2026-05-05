from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import clone_data
from .model import NodeGCN


def train_local_model(
    global_model: NodeGCN,
    local_data,
    owned_train_mask: torch.Tensor,
    lr: float,
    weight_decay: float,
    local_epochs: int,
    device: torch.device,
    poison_loss_weight: float = 1.0,
    return_risk_payload: bool = False,
) -> Dict[str, object]:
    local_model = deepcopy(global_model).to(device)
    local_graph = clone_data(local_data).to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=weight_decay)
    local_mask = owned_train_mask.to(device)

    epoch_losses: List[float] = []
    for _ in range(local_epochs):
        local_model.train()
        optimizer.zero_grad()
        train_edge_index = getattr(local_graph, "train_edge_index", local_graph.edge_index)
        train_edge_weight = getattr(local_graph, "train_edge_weight", getattr(local_graph, "edge_weight", None))
        logits = local_model(local_graph.x, train_edge_index, train_edge_weight)
        per_node_loss = F.cross_entropy(logits[local_mask], local_graph.y[local_mask], reduction="none")
        if hasattr(local_graph, "poisoned_target_mask"):
            poisoned_mask = local_graph.poisoned_target_mask[local_mask].float()
            weights = 1.0 + poisoned_mask * float(max(0.0, poison_loss_weight - 1.0))
            loss = (per_node_loss * weights).mean()
        else:
            loss = per_node_loss.mean()
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.detach().cpu().item()))

    result = {
        "state_dict": {key: value.detach().cpu() for key, value in local_model.state_dict().items()},
        "num_samples": int(local_mask.sum().item()),
        "mean_loss": float(sum(epoch_losses) / len(epoch_losses)),
    }
    if return_risk_payload:
        with torch.no_grad():
            local_model.eval()
            risk_edge_index = getattr(local_graph, "train_edge_index", local_graph.edge_index)
            risk_edge_weight = getattr(local_graph, "train_edge_weight", getattr(local_graph, "edge_weight", None))
            logits, reprs = local_model(local_graph.x, risk_edge_index, risk_edge_weight, return_repr=True)
            node_ids = torch.where(local_mask)[0]
            preds = logits.argmax(dim=-1)
            global_node_ids = (
                local_graph.index_orig[node_ids].detach().cpu().numpy()
                if hasattr(local_graph, "index_orig")
                else node_ids.detach().cpu().numpy()
            )
            is_poisoned = (
                local_graph.poisoned_target_mask[node_ids].detach().cpu().numpy().astype(np.int64)
                if hasattr(local_graph, "poisoned_target_mask")
                else np.zeros(node_ids.numel(), dtype=np.int64)
            )
            repr_keys = sorted(reprs.keys(), key=lambda key: int(key[1:]) if key.startswith("h") else 0)
            if len(repr_keys) >= 3:
                h_nm2_key, h_nm1_key, h_n_key = repr_keys[-3:]
            elif len(repr_keys) == 2:
                h_nm2_key, h_nm1_key, h_n_key = repr_keys[0], repr_keys[0], repr_keys[1]
            else:
                h_nm2_key = h_nm1_key = h_n_key = repr_keys[0]
            h1_key = "h1" if "h1" in reprs else h_nm2_key
            h2_key = "h2" if "h2" in reprs else h_nm1_key
            h3_key = "h3" if "h3" in reprs else h_n_key
            result["risk_payload"] = {
                "client_id": int(getattr(local_graph, "client_id", -1)),
                "local_node_id": node_ids.detach().cpu().numpy(),
                "global_node_id": global_node_ids,
                "is_poisoned": is_poisoned,
                "prediction": preds[node_ids].detach().cpu().numpy(),
                "h1": reprs[h1_key][node_ids].detach().cpu().numpy(),
                "h2": reprs[h2_key][node_ids].detach().cpu().numpy(),
                "h3": reprs[h3_key][node_ids].detach().cpu().numpy(),
                "h_nm2": reprs[h_nm2_key][node_ids].detach().cpu().numpy(),
                "h_nm1": reprs[h_nm1_key][node_ids].detach().cpu().numpy(),
                "h_n": reprs[h_n_key][node_ids].detach().cpu().numpy(),
            }
    return result


def aggregate_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    total_weight = float(sum(weights))
    if total_weight <= 1e-12:
        raise ValueError("Aggregation weights must sum to a positive value.")
    aggregated: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        template = state_dicts[0][key]
        if not torch.is_floating_point(template):
            aggregated[key] = template.clone()
            continue
        aggregated[key] = sum(
            state_dict[key].float() * (weight / total_weight)
            for state_dict, weight in zip(state_dicts, weights)
        )
    return aggregated


def blend_state_dicts(
    anchor_state_dict: Dict[str, torch.Tensor],
    aux_state_dict: Dict[str, torch.Tensor],
    aux_alpha: float,
) -> Dict[str, torch.Tensor]:
    aux_alpha = float(np.clip(aux_alpha, 0.0, 1.0))
    blended: Dict[str, torch.Tensor] = {}
    for key, anchor_value in anchor_state_dict.items():
        aux_value = aux_state_dict[key]
        if not torch.is_floating_point(anchor_value):
            blended[key] = anchor_value.clone()
            continue
        blended[key] = anchor_value.float() * (1.0 - aux_alpha) + aux_value.float() * aux_alpha
    return blended


def flatten_state_delta(
    reference_state_dict: Dict[str, torch.Tensor],
    target_state_dict: Dict[str, torch.Tensor],
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for key, ref_value in reference_state_dict.items():
        target_value = target_state_dict[key]
        if not torch.is_floating_point(ref_value):
            continue
        delta = (target_value.float() - ref_value.float()).reshape(-1).cpu().numpy()
        chunks.append(delta)
    if not chunks:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64, copy=False)


def cosine_similarity_numpy(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))


@torch.no_grad()
def evaluate_clean(
    model: NodeGCN,
    data,
    device: torch.device,
) -> Dict[str, float]:
    eval_data = clone_data(data).to(device)
    model = model.to(device)
    model.eval()
    logits = model(eval_data.x, eval_data.edge_index, getattr(eval_data, "edge_weight", None))
    mask = eval_data.test_mask
    loss = F.cross_entropy(logits[mask], eval_data.y[mask]).item()
    preds = logits.argmax(dim=-1)
    acc = float((preds[mask] == eval_data.y[mask]).float().mean().item())
    return {"test_loss": float(loss), "test_accuracy": acc}
