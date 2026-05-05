from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.utils import subgraph, to_networkx


SPLIT_PROTOCOL = "labeled_nodes_train60_val20_test20_client_local_unlabeled_attack_pool_v1"
LOCAL_SUBGRAPH_CACHE_VERSION = "clean_local_subgraphs_v1"
LOCAL_SUBGRAPH_CACHE_ROOT = Path("outputs/local_subgraph_cache")


def _dataset_key(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def load_planetoid_dataset(root: str, name: str):
    """Load supported node-classification datasets through PyG.

    The training entrypoint uses this helper for all supported node-classification
    datasets, including Planetoid, Coauthor, and Amazon benchmarks.
    """
    raw_name = str(name).strip()
    key = _dataset_key(raw_name)

    planetoid_names = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
    }
    coauthor_names = {
        "coauthorcs": "CS",
        "cs": "CS",
        "coauthorph": "Physics",
        "coauthorphysics": "Physics",
        "physics": "Physics",
    }
    amazon_names = {
        "amzphoto": "Photo",
        "amazonphoto": "Photo",
        "photo": "Photo",
    }

    if key in planetoid_names:
        return Planetoid(root=root, name=planetoid_names[key])
    if key in coauthor_names:
        return Coauthor(root=root, name=coauthor_names[key])
    if key in amazon_names:
        return Amazon(root=root, name=amazon_names[key])
    raise ValueError(f"Unsupported dataset: {raw_name}")


def _split_indices_by_class(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    eligible_indices: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    if eligible_indices is None:
        eligible_indices = np.arange(labels.shape[0], dtype=np.int64)
    eligible_indices = np.asarray(eligible_indices, dtype=np.int64)

    num_classes = int(labels.max()) + 1
    for class_id in range(num_classes):
        class_indices = eligible_indices[labels[eligible_indices] == class_id]
        if len(class_indices) == 0:
            continue
        shuffled = rng.permutation(class_indices)

        num_train = max(1, int(round(len(shuffled) * train_ratio)))
        num_val = max(1, int(round(len(shuffled) * val_ratio)))
        num_train = min(num_train, len(shuffled) - 2) if len(shuffled) >= 3 else min(num_train, len(shuffled))
        remaining = len(shuffled) - num_train
        num_val = min(num_val, max(1, remaining - 1)) if remaining >= 2 else min(num_val, remaining)

        train_slice = shuffled[:num_train]
        val_slice = shuffled[num_train : num_train + num_val]
        test_slice = shuffled[num_train + num_val :]

        train_indices.extend(train_slice.tolist())
        val_indices.extend(val_slice.tolist())
        test_indices.extend(test_slice.tolist())

    return (
        np.array(sorted(train_indices), dtype=np.int64),
        np.array(sorted(val_indices), dtype=np.int64),
        np.array(sorted(test_indices), dtype=np.int64),
    )


def _coerce_1d_bool_mask(mask: torch.Tensor, num_nodes: int) -> torch.Tensor:
    mask = mask.detach().cpu().bool()
    if mask.dim() > 1:
        mask = mask.any(dim=tuple(range(1, mask.dim())))
    return mask[:num_nodes].clone()


def infer_labeled_mask(data: Data) -> torch.Tensor:
    if all(hasattr(data, attr) for attr in ("train_mask", "val_mask", "test_mask")):
        masks = [
            _coerce_1d_bool_mask(getattr(data, attr), data.num_nodes)
            for attr in ("train_mask", "val_mask", "test_mask")
        ]
        labeled_mask = masks[0] | masks[1] | masks[2]
        if int(labeled_mask.sum().item()) > 0:
            return labeled_mask
    return torch.ones(data.num_nodes, dtype=torch.bool)


def refresh_split_indices(data: Data, target_label: int | None = None) -> Data:
    if not hasattr(data, "unlabeled_mask"):
        data.unlabeled_mask = torch.bitwise_not(data.train_mask | data.val_mask | data.test_mask)
    if not hasattr(data, "labeled_mask"):
        data.labeled_mask = data.train_mask | data.val_mask | data.test_mask

    data.idx_train = torch.where(data.train_mask)[0].long()
    data.idx_val = torch.where(data.val_mask)[0].long()
    data.idx_test = torch.where(data.test_mask)[0].long()
    data.idx_unlabeled = torch.where(data.unlabeled_mask)[0].long()

    if target_label is not None and hasattr(data, "clean_y"):
        test_idx = data.idx_test
        if test_idx.numel() > 0:
            non_target_mask = data.clean_y[test_idx] != int(target_label)
            data.idx_atk_test = test_idx[non_target_mask].long()
        else:
            data.idx_atk_test = torch.empty(0, dtype=torch.long, device=test_idx.device)

    if hasattr(data, "index_orig"):
        for attr in ("train", "val", "test", "unlabeled", "atk_test"):
            idx_name = f"idx_{attr}"
            if not hasattr(data, idx_name):
                continue
            idx_value = getattr(data, idx_name)
            setattr(data, f"global_idx_{attr}", data.index_orig[idx_value].long())
    return data


def create_custom_masks(
    data: Data,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Data:
    labels = data.y.cpu().numpy()
    labeled_mask = infer_labeled_mask(data)
    labeled_idx = torch.where(labeled_mask)[0].cpu().numpy()
    train_idx, val_idx, test_idx = _split_indices_by_class(
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        eligible_indices=labeled_idx,
    )

    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.labeled_mask = labeled_mask.clone()
    data.unlabeled_mask = torch.bitwise_not(labeled_mask)
    data.clean_test_mask = test_mask.clone()
    data.attack_test_mask = test_mask.clone()
    data.clean_y = data.y.clone()
    data.split_protocol = SPLIT_PROTOCOL
    data = refresh_split_indices(data)
    return data


def indices_to_mask(indices: Sequence[int] | torch.Tensor, num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    if isinstance(indices, torch.Tensor):
        if indices.numel() > 0:
            mask[indices.long()] = True
        return mask
    if len(indices) > 0:
        mask[torch.tensor(list(indices), dtype=torch.long)] = True
    return mask


def clone_data(data: Data) -> Data:
    cloned = data.clone()
    for attr_name in [
        "clean_y",
        "poisoned_target_mask",
        "owned_train_mask",
        "client_id",
        "triggered_eval_mask",
        "index_orig",
        "clean_test_mask",
        "attack_test_mask",
        "labeled_mask",
        "unlabeled_mask",
        "idx_train",
        "idx_val",
        "idx_test",
        "idx_unlabeled",
        "idx_atk_test",
        "global_idx_train",
        "global_idx_val",
        "global_idx_test",
        "global_idx_unlabeled",
        "global_idx_atk_test",
        "split_protocol",
        "train_edge_index",
        "train_edge_weight",
    ]:
        if hasattr(data, attr_name):
            attr_value = getattr(data, attr_name)
            setattr(cloned, attr_name, attr_value.clone() if hasattr(attr_value, "clone") else attr_value)
    return cloned


def build_local_subgraph(data: Data, node_indices: Sequence[int], client_id: int) -> Data:
    subset = torch.tensor(sorted(int(node) for node in node_indices), dtype=torch.long)
    edge_index, _, edge_mask = subgraph(
        subset,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        return_edge_mask=True,
    )
    local = Data(
        x=data.x[subset].clone(),
        edge_index=edge_index.contiguous(),
        y=data.y[subset].clone(),
    )
    if hasattr(data, "edge_weight") and getattr(data, "edge_weight", None) is not None:
        local.edge_weight = data.edge_weight[edge_mask].clone()
    else:
        local.edge_weight = torch.ones(local.edge_index.size(1), dtype=torch.float32)
    local.clean_y = local.y.clone()
    if hasattr(data, "index_orig"):
        local.index_orig = data.index_orig[subset].clone()
    else:
        local.index_orig = subset.clone()
    local.client_id = int(client_id)
    return local


def expand_nodes_with_one_hop(
    data: Data,
    seed_nodes: Sequence[int],
    max_total_nodes: int | None = None,
    seed: int = 0,
) -> List[int]:
    if not seed_nodes:
        return []
    edge_pairs = data.edge_index.detach().cpu().t().tolist()
    node_set: Set[int] = {int(node) for node in seed_nodes}
    boundary: Set[int] = set()
    for src, dst in edge_pairs:
        if int(src) in node_set:
            boundary.add(int(dst))
        if int(dst) in node_set:
            boundary.add(int(src))
    boundary = boundary - node_set
    if max_total_nodes is None or len(node_set) + len(boundary) <= max_total_nodes:
        node_set.update(boundary)
        return sorted(node_set)

    budget = max(0, int(max_total_nodes) - len(node_set))
    if budget <= 0:
        return sorted(node_set)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(sorted(boundary), size=min(budget, len(boundary)), replace=False).tolist()
    node_set.update(int(node) for node in sampled)
    return sorted(node_set)


def _community_node_lists(graph: nx.Graph) -> List[List[int]]:
    communities = nx.algorithms.community.louvain_communities(graph, seed=0)
    return [sorted(int(node) for node in community) for community in communities]


def _assign_communities_to_clients(
    community_lists: List[List[int]],
    num_clients: int,
) -> List[List[int]]:
    client_nodes: List[List[int]] = [[] for _ in range(num_clients)]
    client_loads = [0 for _ in range(num_clients)]

    for community in sorted(community_lists, key=len, reverse=True):
        target_client = int(np.argmin(client_loads))
        client_nodes[target_client].extend(community)
        client_loads[target_client] += len(community)

    for bucket in client_nodes:
        bucket.sort()
    return client_nodes


def louvain_partition_nodes(
    data: Data,
    num_clients: int,
    seed: int,
    min_size: int = 20,
) -> Tuple[List[List[int]], List[List[int]]]:
    del seed
    base_graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
    all_communities = _community_node_lists(base_graph)

    train_communities: List[List[int]] = []
    train_mask_np = data.train_mask.cpu().numpy().astype(bool)
    for community in all_communities:
        train_nodes = [node for node in community if train_mask_np[node]]
        if train_nodes:
            train_communities.append(train_nodes)

    client_node_indices = _assign_communities_to_clients(train_communities, num_clients=num_clients)

    merged = True
    while merged:
        merged = False
        small_clients = [idx for idx, bucket in enumerate(client_node_indices) if 0 < len(bucket) < min_size]
        if not small_clients:
            break
        for client_id in small_clients:
            donor_candidates = [idx for idx, bucket in enumerate(client_node_indices) if idx != client_id and len(bucket) > min_size]
            if not donor_candidates:
                continue
            donor_id = max(donor_candidates, key=lambda idx: len(client_node_indices[idx]))
            take = max(1, min(min_size - len(client_node_indices[client_id]), len(client_node_indices[donor_id]) - min_size))
            moved = client_node_indices[donor_id][:take]
            client_node_indices[donor_id] = client_node_indices[donor_id][take:]
            client_node_indices[client_id].extend(moved)
            client_node_indices[client_id].sort()
            merged = True

    if any(len(bucket) == 0 for bucket in client_node_indices):
        non_empty = [bucket for bucket in client_node_indices if bucket]
        if not non_empty:
            raise RuntimeError("Louvain partition produced no training nodes.")
        for client_id, bucket in enumerate(client_node_indices):
            if bucket:
                continue
            donor_id = max(range(len(client_node_indices)), key=lambda idx: len(client_node_indices[idx]))
            donor_bucket = client_node_indices[donor_id]
            split_size = max(1, len(donor_bucket) // 2)
            client_node_indices[client_id] = donor_bucket[:split_size]
            client_node_indices[donor_id] = donor_bucket[split_size:]

    client_community_lists: List[List[int]] = [[] for _ in range(num_clients)]
    for community_idx, community in enumerate(train_communities):
        community_set = set(community)
        best_client_id = max(
            range(num_clients),
            key=lambda client_id: len(community_set.intersection(client_node_indices[client_id])),
        )
        client_community_lists[best_client_id].append(community_idx)

    for bucket in client_node_indices:
        bucket.sort()
    return client_node_indices, train_communities


def _kmeans_assignments(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    max_iters: int = 50,
    chunk_size: int = 1024,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features for k-means, got shape {features.shape}.")
    num_samples = int(features.shape[0])
    if num_samples == 0:
        return np.empty(0, dtype=np.int64)

    num_clusters = max(1, min(int(num_clusters), num_samples))
    rng = np.random.default_rng(seed)
    work_features = np.asarray(features, dtype=np.float32)
    init_ids = rng.choice(num_samples, size=num_clusters, replace=False)
    centroids = work_features[init_ids].astype(np.float32, copy=True)
    assignments = np.zeros(num_samples, dtype=np.int64)
    chunk_size = max(1, int(chunk_size))

    for _ in range(max(1, int(max_iters))):
        centroid_norms = (centroids * centroids).sum(axis=1)
        new_assignments = np.empty(num_samples, dtype=np.int64)
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            feature_chunk = work_features[start:end]
            chunk_norms = (feature_chunk * feature_chunk).sum(axis=1, keepdims=True)
            distances = chunk_norms + centroid_norms[None, :] - 2.0 * (feature_chunk @ centroids.T)
            new_assignments[start:end] = distances.argmin(axis=1).astype(np.int64, copy=False)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        for cluster_id in range(num_clusters):
            mask = assignments == cluster_id
            if np.any(mask):
                centroids[cluster_id] = work_features[mask].mean(axis=0)
            else:
                refill_id = int(rng.integers(0, num_samples))
                centroids[cluster_id] = work_features[refill_id]

    return assignments


def _safe_dirichlet_split(
    rng: np.random.Generator,
    source_indices: np.ndarray,
    num_clients: int,
    alpha: float,
) -> List[np.ndarray]:
    if len(source_indices) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(num_clients)]

    shuffled = np.array(source_indices, copy=True)
    rng.shuffle(shuffled)
    proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
    split_points = (np.cumsum(proportions)[:-1] * len(shuffled)).astype(int)
    return [chunk.astype(np.int64, copy=False) for chunk in np.split(shuffled, split_points)]


def dirichlet_partition_nodes(
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 20,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_indices_np = train_indices.cpu().numpy()
    num_classes = int(labels_np.max()) + 1

    while True:
        client_buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = train_indices_np[labels_np[train_indices_np] == class_id]
            if len(class_indices) == 0:
                continue
            splits = _safe_dirichlet_split(
                rng=rng,
                source_indices=class_indices,
                num_clients=num_clients,
                alpha=alpha,
            )
            for client_id, split in enumerate(splits):
                client_buckets[client_id].extend(split.tolist())

        client_sizes = [len(bucket) for bucket in client_buckets]
        if min(client_sizes) >= min_size:
            break

    for bucket in client_buckets:
        bucket.sort()
    return client_buckets


def feature_skew_partition_nodes(
    features: torch.Tensor,
    train_indices: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 20,
    num_feature_clusters: int | None = None,
) -> Tuple[List[List[int]], Dict[int, int]]:
    rng = np.random.default_rng(seed)
    train_indices_np = train_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    if len(train_indices_np) == 0:
        raise RuntimeError("Feature-skew partition requires at least one training node.")

    feature_matrix = features[train_indices].detach().cpu().float().numpy()
    requested_clusters = num_feature_clusters if num_feature_clusters is not None else max(2, min(num_clients, 8))
    cluster_ids_local = _kmeans_assignments(
        features=feature_matrix,
        num_clusters=int(requested_clusters),
        seed=seed,
    )

    while True:
        client_buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for cluster_id in sorted(set(cluster_ids_local.tolist())):
            cluster_indices = train_indices_np[cluster_ids_local == cluster_id]
            splits = _safe_dirichlet_split(
                rng=rng,
                source_indices=cluster_indices,
                num_clients=num_clients,
                alpha=alpha,
            )
            for client_id, split in enumerate(splits):
                client_buckets[client_id].extend(split.tolist())

        client_sizes = [len(bucket) for bucket in client_buckets]
        if min(client_sizes) >= min_size:
            break

    for bucket in client_buckets:
        bucket.sort()
    feature_cluster_map = {
        int(global_idx): int(cluster_id)
        for global_idx, cluster_id in zip(train_indices_np.tolist(), cluster_ids_local.tolist())
    }
    return client_buckets, feature_cluster_map


def iid_partition_nodes(
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    num_clients: int,
    seed: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_indices_np = train_indices.cpu().numpy()
    num_classes = int(labels_np.max()) + 1
    client_buckets: List[List[int]] = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = train_indices_np[labels_np[train_indices_np] == class_id]
        rng.shuffle(class_indices)
        client_order = rng.permutation(num_clients)
        for offset, node_idx in enumerate(class_indices.tolist()):
            client_buckets[int(client_order[offset % num_clients])].append(int(node_idx))

    for bucket in client_buckets:
        bucket.sort()
    return client_buckets


def dirichlet_partition_graph(
    labels: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 20,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    all_indices_np = np.arange(labels.size(0))
    num_classes = int(labels_np.max()) + 1

    while True:
        client_buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = all_indices_np[labels_np == class_id]
            if len(class_indices) == 0:
                continue
            splits = _safe_dirichlet_split(
                rng=rng,
                source_indices=class_indices,
                num_clients=num_clients,
                alpha=alpha,
            )
            for client_id, split in enumerate(splits):
                client_buckets[client_id].extend(int(idx) for idx in split.tolist())
        client_sizes = [len(bucket) for bucket in client_buckets]
        if min(client_sizes) >= min_size:
            break

    for bucket in client_buckets:
        bucket.sort()
    return client_buckets


def split_local_masks(
    data: Data,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Data:
    labels = data.y.cpu().numpy()
    train_idx, val_idx, test_idx = _split_indices_by_class(
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    num_nodes = data.num_nodes
    data.train_mask = indices_to_mask(train_idx, num_nodes)
    data.val_mask = indices_to_mask(val_idx, num_nodes)
    data.test_mask = indices_to_mask(test_idx, num_nodes)
    data.labeled_mask = data.train_mask | data.val_mask | data.test_mask
    data.unlabeled_mask = torch.bitwise_not(data.labeled_mask)
    data.clean_test_mask = data.test_mask.clone()
    data.attack_test_mask = data.test_mask.clone()

    train_edge_index, _, train_edge_mask = subgraph(
        torch.bitwise_not(data.test_mask),
        data.edge_index,
        relabel_nodes=False,
        num_nodes=data.num_nodes,
        return_edge_mask=True,
    )
    data.train_edge_index = train_edge_index.contiguous()
    if hasattr(data, "edge_weight") and getattr(data, "edge_weight", None) is not None:
        data.train_edge_weight = data.edge_weight[train_edge_mask].clone()
    else:
        data.train_edge_weight = torch.ones(data.train_edge_index.size(1), dtype=torch.float32)
    data.poisoned_target_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.owned_train_mask = data.train_mask.clone()
    data.split_protocol = SPLIT_PROTOCOL
    data = refresh_split_indices(data)
    return data


def apply_global_local_masks(
    data: Data,
    owned_global_train_nodes: Sequence[int],
    full_data: Data,
) -> Data:
    global_index = data.index_orig.long()
    if hasattr(full_data, "index_orig"):
        owned_set = {int(full_data.index_orig[int(node)].item()) for node in owned_global_train_nodes}
        full_global_to_local = {int(global_id): local_idx for local_idx, global_id in enumerate(full_data.index_orig.tolist())}
        full_local_index = torch.tensor(
            [int(full_global_to_local[int(node)]) for node in global_index.tolist()],
            dtype=torch.long,
        )
        data.val_mask = full_data.val_mask[full_local_index].clone()
        data.test_mask = full_data.test_mask[full_local_index].clone()
    else:
        owned_set = {int(node) for node in owned_global_train_nodes}
        data.val_mask = full_data.val_mask[global_index].clone()
        data.test_mask = full_data.test_mask[global_index].clone()
    owned_local_train = [idx for idx, node in enumerate(global_index.tolist()) if int(node) in owned_set]
    data.train_mask = indices_to_mask(owned_local_train, data.num_nodes)
    data.labeled_mask = data.train_mask | data.val_mask | data.test_mask
    data.unlabeled_mask = torch.bitwise_not(data.labeled_mask)
    data.clean_test_mask = data.test_mask.clone()
    data.attack_test_mask = data.test_mask.clone()
    train_edge_index, _, train_edge_mask = subgraph(
        torch.bitwise_not(data.test_mask),
        data.edge_index,
        relabel_nodes=False,
        num_nodes=data.num_nodes,
        return_edge_mask=True,
    )
    data.train_edge_index = train_edge_index.contiguous()
    if hasattr(data, "edge_weight") and getattr(data, "edge_weight", None) is not None:
        data.train_edge_weight = data.edge_weight[train_edge_mask].clone()
    else:
        data.train_edge_weight = torch.ones(data.train_edge_index.size(1), dtype=torch.float32)
    data.poisoned_target_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.owned_train_mask = data.train_mask.clone()
    data.split_protocol = SPLIT_PROTOCOL
    data = refresh_split_indices(data)
    return data


def _sanitize_cache_component(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    safe_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def resolve_local_subgraph_cache_dir(
    partition_metadata_path: str | Path | None,
    partition_name: str,
    seed: int,
    num_clients: int,
    num_nodes: int,
) -> Path:
    if partition_metadata_path:
        namespace = Path(str(partition_metadata_path)).stem
    else:
        namespace = (
            f"{_sanitize_cache_component(partition_name)}"
            f"_seed{int(seed)}_clients{int(num_clients)}_nodes{int(num_nodes)}"
        )
    return LOCAL_SUBGRAPH_CACHE_ROOT / LOCAL_SUBGRAPH_CACHE_VERSION / _sanitize_cache_component(namespace)


def build_cached_clean_local_subgraphs(
    source_data: Data,
    client_node_indices: Sequence[Sequence[int]],
    full_data: Data,
    seed: int,
    partition_metadata_path: str | Path | None,
    partition_name: str,
) -> List[Data]:
    cache_dir = resolve_local_subgraph_cache_dir(
        partition_metadata_path=partition_metadata_path,
        partition_name=partition_name,
        seed=seed,
        num_clients=len(client_node_indices),
        num_nodes=int(source_data.num_nodes),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    clean_local_graphs: List[Data] = []
    for client_id, node_indices in enumerate(client_node_indices):
        normalized_seed_nodes = [int(node) for node in node_indices]
        owned_global_train_nodes = (
            source_data.index_orig[torch.tensor(normalized_seed_nodes, dtype=torch.long)].cpu().tolist()
        )
        cache_path = cache_dir / f"client_{int(client_id):03d}.pt"
        cached_graph = None
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                if (
                    isinstance(payload, dict)
                    and payload.get("cache_version") == LOCAL_SUBGRAPH_CACHE_VERSION
                    and int(payload.get("client_id", -1)) == int(client_id)
                    and [int(node) for node in payload.get("train_node_indices", [])] == normalized_seed_nodes
                    and [int(node) for node in payload.get("owned_global_train_nodes", [])] == owned_global_train_nodes
                ):
                    graph_payload = payload.get("graph")
                    if isinstance(graph_payload, Data):
                        cached_graph = clone_data(graph_payload)
            except Exception:
                cached_graph = None
        if cached_graph is None:
            local_nodes = expand_nodes_with_one_hop(
                source_data,
                seed_nodes=normalized_seed_nodes,
                max_total_nodes=max(len(normalized_seed_nodes) * 4, len(normalized_seed_nodes) + 200),
                seed=int(seed) + int(client_id),
            )
            local_graph = build_local_subgraph(source_data, node_indices=local_nodes, client_id=int(client_id))
            local_graph = apply_global_local_masks(
                data=local_graph,
                owned_global_train_nodes=owned_global_train_nodes,
                full_data=full_data,
            )
            torch.save(
                {
                    "cache_version": LOCAL_SUBGRAPH_CACHE_VERSION,
                    "client_id": int(client_id),
                    "partition_name": str(partition_name),
                    "seed": int(seed),
                    "train_node_indices": normalized_seed_nodes,
                    "owned_global_train_nodes": [int(node) for node in owned_global_train_nodes],
                    "graph": clone_data(local_graph),
                },
                cache_path,
            )
            cached_graph = local_graph
        clean_local_graphs.append(cached_graph)
    return clean_local_graphs


def _global_indices_for_mask(data: Data, mask: torch.Tensor) -> List[int]:
    local_idx = torch.where(mask)[0].long()
    if hasattr(data, "index_orig"):
        return [int(node) for node in data.index_orig[local_idx].detach().cpu().tolist() if int(node) >= 0]
    return [int(node) for node in local_idx.detach().cpu().tolist()]


def _local_indices_for_mask(mask: torch.Tensor) -> List[int]:
    return [int(node) for node in torch.where(mask)[0].detach().cpu().tolist()]


def attach_client_split_metadata(
    partition_metadata: Dict[str, object],
    local_graphs: Sequence[Data],
    target_label: int | None = None,
) -> Dict[str, object]:
    partition_metadata["split_protocol"] = SPLIT_PROTOCOL
    partition_metadata["metric_protocol"] = {
        "acc": "mean client clean accuracy on full idx_test",
        "asr": "mean malicious-client target-class rate on idx_test nodes whose clean label is not target",
        "transfer_asr": "mean benign-client target-class rate after applying malicious trigger to benign idx_test non-target nodes",
    }
    clients = partition_metadata.get("clients", [])
    for graph in local_graphs:
        refresh_split_indices(graph, target_label=target_label)
        client_id = int(getattr(graph, "client_id", len(clients)))
        if client_id >= len(clients):
            continue
        client_info = clients[client_id]
        atk_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.test_mask.device)
        if target_label is not None:
            atk_mask = graph.test_mask & (graph.clean_y != int(target_label))
        client_info.update(
            {
                "idx_train": _global_indices_for_mask(graph, graph.train_mask),
                "idx_val": _global_indices_for_mask(graph, graph.val_mask),
                "idx_test": _global_indices_for_mask(graph, graph.test_mask),
                "idx_unlabeled": _global_indices_for_mask(graph, graph.unlabeled_mask),
                "idx_atk_test": _global_indices_for_mask(graph, atk_mask),
                "idx_train_local": _local_indices_for_mask(graph.train_mask),
                "idx_val_local": _local_indices_for_mask(graph.val_mask),
                "idx_test_local": _local_indices_for_mask(graph.test_mask),
                "idx_unlabeled_local": _local_indices_for_mask(graph.unlabeled_mask),
                "idx_atk_test_local": _local_indices_for_mask(atk_mask),
                "num_train_nodes": int(graph.train_mask.sum().item()),
                "num_val_nodes": int(graph.val_mask.sum().item()),
                "num_test_nodes": int(graph.test_mask.sum().item()),
                "num_unlabeled_nodes": int(graph.unlabeled_mask.sum().item()),
                "num_atk_test_nodes": int(atk_mask.sum().item()),
            }
        )
    return partition_metadata


def build_partition_metadata(
    data: Data,
    client_node_indices: List[List[int]],
    partition_name: str,
    community_lists: List[List[int]] | None = None,
    feature_cluster_map: Dict[int, int] | None = None,
    partition_params: Dict[str, object] | None = None,
) -> Dict[str, object]:
    clients = []
    node_to_community: Dict[int, int] = {}
    if community_lists is not None:
        for community_id, nodes in enumerate(community_lists):
            for node in nodes:
                node_to_community[int(node)] = int(community_id)

    for client_id, node_indices in enumerate(client_node_indices):
        label_hist = Counter(data.y[node_indices].tolist())
        community_hist = Counter(node_to_community.get(int(node), -1) for node in node_indices)
        payload = {
            "client_id": client_id,
            "train_node_indices": node_indices,
            "num_nodes": len(node_indices),
            "label_hist": {str(key): int(value) for key, value in sorted(label_hist.items())},
            "community_hist": {str(key): int(value) for key, value in sorted(community_hist.items()) if key >= 0},
        }
        if feature_cluster_map is not None:
            feature_hist = Counter(feature_cluster_map.get(int(node), -1) for node in node_indices)
            payload["feature_cluster_hist"] = {
                str(key): int(value)
                for key, value in sorted(feature_hist.items())
                if key >= 0
            }
        clients.append(payload)

    payload: Dict[str, object] = {
        "split_protocol": SPLIT_PROTOCOL,
        "partition_name": partition_name,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "labeled_indices": torch.where(getattr(data, "labeled_mask", data.train_mask | data.val_mask | data.test_mask))[0].tolist(),
        "unlabeled_indices": torch.where(getattr(data, "unlabeled_mask", torch.bitwise_not(data.train_mask | data.val_mask | data.test_mask)))[0].tolist(),
        "train_indices": torch.where(data.train_mask)[0].tolist(),
        "val_indices": torch.where(data.val_mask)[0].tolist(),
        "test_indices": torch.where(data.test_mask)[0].tolist(),
        "clients": clients,
    }
    if partition_params:
        payload["partition_params"] = dict(partition_params)
    if community_lists is not None:
        payload["communities"] = [
            {"community_id": idx, "train_node_indices": nodes, "num_nodes": len(nodes)}
            for idx, nodes in enumerate(community_lists)
        ]
    if feature_cluster_map is not None:
        payload["feature_clusters"] = [
            {"node_index": int(node_index), "cluster_id": int(cluster_id)}
            for node_index, cluster_id in sorted(feature_cluster_map.items())
        ]
    return payload


def apply_partition_metadata(data: Data, partition_metadata: Dict[str, object]) -> Data:
    data.train_mask = indices_to_mask(partition_metadata["train_indices"], data.num_nodes)
    data.val_mask = indices_to_mask(partition_metadata["val_indices"], data.num_nodes)
    data.test_mask = indices_to_mask(partition_metadata["test_indices"], data.num_nodes)
    if "labeled_indices" in partition_metadata:
        data.labeled_mask = indices_to_mask(partition_metadata["labeled_indices"], data.num_nodes)
    else:
        data.labeled_mask = data.train_mask | data.val_mask | data.test_mask
    if "unlabeled_indices" in partition_metadata:
        data.unlabeled_mask = indices_to_mask(partition_metadata["unlabeled_indices"], data.num_nodes)
    else:
        data.unlabeled_mask = torch.bitwise_not(data.labeled_mask)
    data.clean_test_mask = data.test_mask.clone()
    data.attack_test_mask = data.test_mask.clone()
    data.clean_y = data.y.clone()
    data.split_protocol = str(partition_metadata.get("split_protocol", SPLIT_PROTOCOL))
    data = refresh_split_indices(data)
    return data
