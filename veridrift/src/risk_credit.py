from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def robust_positive_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-8, 1.4826 * mad)
    robust_z = (values - median) / scale
    return 1.0 - np.exp(-np.clip(robust_z, a_min=0.0, a_max=None))


def robust_zscore_array(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    scale = 1.4826 * mad
    normalized = (values - median) / scale
    return np.clip(normalized, a_min=-5.0, a_max=5.0)


def safe_cosine_distance(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_norm = np.linalg.norm(lhs, axis=1)
    rhs_norm = np.linalg.norm(rhs, axis=1)
    denom = np.clip(lhs_norm * rhs_norm, 1e-12, None)
    cosine_sim = np.sum(lhs * rhs, axis=1) / denom
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    return 1.0 - cosine_sim


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return features / norms


def compute_cross_layer_metrics(reprs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    h_nm2 = reprs.get("h_nm2", reprs["h1"])
    h_nm1 = reprs.get("h_nm1", reprs["h2"])
    h_n = reprs.get("h_n", reprs["h3"])
    delta2 = h_n - h_nm1
    delta1 = h_nm1 - h_nm2
    return {
        "delta2": delta2,
        "d2": np.linalg.norm(delta2, axis=1),
        "turn12": safe_cosine_distance(delta1, delta2),
    }


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def mode_label_and_ratio(labels: np.ndarray) -> Tuple[int, float]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return -1, 0.0
    unique_labels, counts = np.unique(labels, return_counts=True)
    best_idx = int(np.argmax(counts))
    return int(unique_labels[best_idx]), float(counts[best_idx] / labels.size)


def build_direction_prototype(
    client_rows: Sequence[Dict[str, object]],
    direction_by_client: Dict[int, np.ndarray],
    candidate_fraction: float,
    descending: bool,
    exclude_client_id: int | None = None,
) -> np.ndarray:
    sorted_rows = sorted(client_rows, key=lambda item: float(item["highrisk_level"]), reverse=descending)
    if exclude_client_id is not None:
        sorted_rows = [row for row in sorted_rows if int(row["client_id"]) != int(exclude_client_id)]
    candidate_count = max(2, int(np.ceil(len(client_rows) * candidate_fraction)))
    selected = sorted_rows[:candidate_count]
    dim = next(iter(direction_by_client.values())).shape[0]
    prototype = np.zeros(dim, dtype=np.float64)
    total_weight = 0.0
    for row in selected:
        direction = direction_by_client[int(row["client_id"])]
        weight = max(0.1, float(row["highrisk_level"]))
        prototype += weight * direction
        total_weight += weight
    if total_weight <= 1e-8:
        return np.zeros(dim, dtype=np.float64)
    prototype = prototype / total_weight
    norm = float(np.linalg.norm(prototype))
    if norm <= 1e-8:
        return np.zeros(dim, dtype=np.float64)
    return prototype / norm


def summarize_detection(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        f"{prefix}_client_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_client_auc": float(auc),
        f"{prefix}_client_precision": float(precision),
        f"{prefix}_client_recall": float(recall),
        f"{prefix}_client_f1": float(f1),
    }


def compute_client_risk_features_from_local_results(
    local_results: Sequence[Dict[str, object]],
    malicious_client_ids: Sequence[int] | None = None,
    highrisk_quantile: float = 0.90,
    suspicious_candidate_fraction: float = 0.30,
    benign_candidate_fraction: float = 0.40,
    highrisk_history_by_client_id: Dict[int, List[float]] | None = None,
    early_history_by_client_id: Dict[int, List[int]] | None = None,
    early_threshold_mad_mult: float = 0.5,
    confirmed_threshold_mad_mult: float = 1.0,
    confirmed_window: int = 3,
    confirmed_patience: int = 2,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    malicious_set = {int(client_id) for client_id in (malicious_client_ids or [])}
    node_rows: List[Dict[str, object]] = []
    client_rows: List[Dict[str, object]] = []
    direction_by_client: Dict[int, np.ndarray] = {}
    top_predictions_by_client: Dict[int, np.ndarray] = {}
    if highrisk_history_by_client_id is None:
        highrisk_history_by_client_id = {}
    if early_history_by_client_id is None:
        early_history_by_client_id = {}

    for local_result in local_results:
        risk_payload = local_result.get("risk_payload")
        if not risk_payload:
            continue
        client_id = int(risk_payload["client_id"])
        is_malicious_client = int(client_id in malicious_set)
        if len(risk_payload["local_node_id"]) == 0:
            continue
        client_reprs = {
            "h1": np.stack(risk_payload["h1"], axis=0),
            "h2": np.stack(risk_payload["h2"], axis=0),
            "h3": np.stack(risk_payload["h3"], axis=0),
            "h_nm2": np.stack(risk_payload.get("h_nm2", risk_payload["h1"]), axis=0),
            "h_nm1": np.stack(risk_payload.get("h_nm1", risk_payload["h2"]), axis=0),
            "h_n": np.stack(risk_payload.get("h_n", risk_payload["h3"]), axis=0),
        }
        drift = compute_cross_layer_metrics(client_reprs)
        d2_score = robust_positive_score(drift["d2"])
        turn12_score = robust_positive_score(drift["turn12"])
        node_risk = 0.5 * (d2_score + turn12_score)
        delta2_unit_vectors = normalize_rows(drift["delta2"])

        for index in range(len(risk_payload["local_node_id"])):
            node_rows.append(
                {
                    "client_id": client_id,
                    "local_node_id": int(risk_payload["local_node_id"][index]),
                    "global_node_id": int(risk_payload["global_node_id"][index]),
                    "is_malicious_client": is_malicious_client,
                    "is_poisoned": int(risk_payload["is_poisoned"][index]),
                    "prediction": int(risk_payload["prediction"][index]),
                    "node_risk": float(node_risk[index]),
                    "d2": float(drift["d2"][index]),
                    "turn12": float(drift["turn12"][index]),
                }
            )

        highrisk_level = float(np.quantile(node_risk, highrisk_quantile))
        highrisk_indices = np.where(node_risk >= highrisk_level)[0]
        if len(highrisk_indices) == 0:
            highrisk_indices = np.asarray([int(np.argmax(node_risk))], dtype=np.int64)

        history = highrisk_history_by_client_id.get(int(client_id), [])
        history_baseline = float(np.median(history[-2:])) if history[-2:] else float(highrisk_level)
        highrisk_growth = max(0.0, float(highrisk_level - history_baseline))
        highrisk_history_by_client_id.setdefault(int(client_id), []).append(float(highrisk_level))

        direction = delta2_unit_vectors[highrisk_indices].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        direction_by_client[int(client_id)] = (
            np.zeros(delta2_unit_vectors.shape[1], dtype=np.float64)
            if norm <= 1e-8
            else direction / norm
        )

        predictions = np.asarray(risk_payload["prediction"], dtype=np.int64)
        is_poisoned = np.asarray(risk_payload["is_poisoned"], dtype=np.int64)
        top_predictions = predictions[highrisk_indices]
        top_predictions_by_client[int(client_id)] = top_predictions
        top_pred_mode_label, top_pred_mode_ratio = mode_label_and_ratio(top_predictions)
        client_rows.append(
            {
                "client_id": int(client_id),
                "is_malicious": int(client_id in malicious_set),
                "num_nodes": int(len(risk_payload["local_node_id"])),
                "num_poisoned_nodes": int(is_poisoned.sum()),
                "highrisk_level": float(highrisk_level),
                "highrisk_growth": float(highrisk_growth),
                "top_pred_mode_label": int(top_pred_mode_label),
                "top_pred_mode_ratio": float(top_pred_mode_ratio),
            }
        )

    if not client_rows:
        return [], [], {}

    suspicious_count = max(2, int(np.ceil(len(client_rows) * suspicious_candidate_fraction)))
    suspicious_client_rows = sorted(client_rows, key=lambda item: float(item["highrisk_level"]), reverse=True)[:suspicious_count]
    suspicious_labels: List[int] = []
    for row in suspicious_client_rows:
        suspicious_labels.extend(top_predictions_by_client[int(row["client_id"])].tolist())
    suspicious_target_label, suspicious_target_ratio = mode_label_and_ratio(np.asarray(suspicious_labels, dtype=np.int64))

    for row in client_rows:
        top_preds = top_predictions_by_client[int(row["client_id"])]
        label_alignment = 0.0 if suspicious_target_label < 0 or top_preds.size == 0 else float(np.mean(top_preds == suspicious_target_label))
        row["suspicious_target_label"] = int(suspicious_target_label)
        row["suspicious_target_ratio"] = float(suspicious_target_ratio)
        row["label_alignment"] = float(label_alignment)

    for row in client_rows:
        client_id = int(row["client_id"])
        suspicious_direction = build_direction_prototype(
            client_rows=client_rows,
            direction_by_client=direction_by_client,
            candidate_fraction=suspicious_candidate_fraction,
            descending=True,
            exclude_client_id=client_id,
        )
        benign_direction = build_direction_prototype(
            client_rows=client_rows,
            direction_by_client=direction_by_client,
            candidate_fraction=benign_candidate_fraction,
            descending=False,
            exclude_client_id=client_id,
        )
        suspicious_similarity = cosine_similarity(direction_by_client[client_id], suspicious_direction)
        benign_similarity = cosine_similarity(direction_by_client[client_id], benign_direction)
        row["consensus_margin"] = float(suspicious_similarity - benign_similarity)

    highrisk_level_rz = robust_zscore_array([row["highrisk_level"] for row in client_rows])
    consensus_margin_rz = robust_zscore_array([row["consensus_margin"] for row in client_rows])
    for index, row in enumerate(client_rows):
        row["highrisk_level_rz"] = float(highrisk_level_rz[index])
        row["consensus_margin_rz"] = float(consensus_margin_rz[index])
        row["risk_score"] = float(row["highrisk_level"])

    risk_scores = np.asarray([row["risk_score"] for row in client_rows], dtype=np.float64)
    median = float(np.median(risk_scores))
    mad = float(np.median(np.abs(risk_scores - median)))
    scale = 1.4826 * mad
    early_threshold = median + early_threshold_mad_mult * scale
    confirmed_threshold = median + confirmed_threshold_mad_mult * scale
    ranked_rows = sorted(client_rows, key=lambda item: float(item["risk_score"]), reverse=True)
    rank_map = {int(row["client_id"]): rank for rank, row in enumerate(ranked_rows, start=1)}

    for row in client_rows:
        client_id = int(row["client_id"])
        early_flag = int(float(row["risk_score"]) > early_threshold)
        confirmed_raw_flag = int(float(row["risk_score"]) > confirmed_threshold)
        history = early_history_by_client_id.setdefault(client_id, [])
        history.append(early_flag)
        recent = history[-confirmed_window:]
        confirmed_flag = int(confirmed_raw_flag == 1 or sum(recent) >= confirmed_patience)
        row["early_threshold"] = float(early_threshold)
        row["confirmed_threshold"] = float(confirmed_threshold)
        row["early_flag"] = int(early_flag)
        row["confirmed_raw_flag"] = int(confirmed_raw_flag)
        row["confirmed_flag"] = int(confirmed_flag)
        row["risk_rank"] = int(rank_map[client_id])

    summary = {
        "mean_malicious_risk": float(np.mean(risk_scores[[i for i,row in enumerate(client_rows) if int(row['is_malicious']) == 1]])) if any(int(row["is_malicious"]) == 1 for row in client_rows) else float("nan"),
        "mean_benign_risk": float(np.mean(risk_scores[[i for i,row in enumerate(client_rows) if int(row['is_malicious']) == 0]])) if any(int(row["is_malicious"]) == 0 for row in client_rows) else float("nan"),
    }
    return node_rows, client_rows, summary


class OnlineRiskCreditTracker:
    def __init__(
        self,
        output_dir: Path,
        malicious_client_ids: Sequence[int] | None = None,
        highrisk_quantile: float = 0.90,
        suspicious_candidate_fraction: float = 0.30,
        benign_candidate_fraction: float = 0.40,
        early_threshold_mad_mult: float = 0.5,
        confirmed_threshold_mad_mult: float = 1.0,
        confirmed_window: int = 3,
        confirmed_patience: int = 2,
    ) -> None:
        self.output_dir = output_dir
        self.malicious_client_ids = {int(client_id) for client_id in (malicious_client_ids or [])}
        self.highrisk_quantile = float(highrisk_quantile)
        self.suspicious_candidate_fraction = float(suspicious_candidate_fraction)
        self.benign_candidate_fraction = float(benign_candidate_fraction)
        self.early_threshold_mad_mult = float(early_threshold_mad_mult)
        self.confirmed_threshold_mad_mult = float(confirmed_threshold_mad_mult)
        self.confirmed_window = int(confirmed_window)
        self.confirmed_patience = int(confirmed_patience)

        self.highrisk_history_by_client_id: Dict[int, List[float]] = {}
        self.early_history_by_client_id: Dict[int, List[int]] = {}
        self.round_summary_rows: List[Dict[str, object]] = []
        self.latest_node_rows: List[Dict[str, object]] = []
        self.latest_client_rows: List[Dict[str, object]] = []
        self.latest_summary_row: Dict[str, object] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.node_path = self.output_dir / "node_risk_by_round.csv"
        self.client_path = self.output_dir / "client_risk_by_round.csv"
        self.summary_path = self.output_dir / "round_detection_summary.csv"

        self.node_fields = [
            "round",
            "client_id",
            "local_node_id",
            "global_node_id",
            "is_malicious_client",
            "is_poisoned",
            "prediction",
            "node_risk",
            "d2",
            "turn12",
        ]
        self.client_fields = [
            "round",
            "client_id",
            "is_malicious",
            "num_nodes",
            "num_poisoned_nodes",
            "highrisk_level",
            "highrisk_growth",
            "consensus_margin",
            "label_alignment",
            "risk_score",
            "early_threshold",
            "confirmed_threshold",
            "early_flag",
            "confirmed_raw_flag",
            "confirmed_flag",
            "risk_rank",
            "top_pred_mode_label",
            "top_pred_mode_ratio",
            "suspicious_target_label",
            "suspicious_target_ratio",
            "highrisk_level_rz",
            "consensus_margin_rz",
        ]
        self.summary_fields = [
            "round",
            "mean_malicious_risk",
            "mean_benign_risk",
            "malicious_risk_gap",
            "topk_client_ids",
            "topk_hit_count",
            "topk_recall",
            "avg_malicious_rank",
            "best_malicious_rank",
            "worst_malicious_rank",
            "early_client_accuracy",
            "early_client_auc",
            "early_client_precision",
            "early_client_recall",
            "early_client_f1",
            "confirmed_client_accuracy",
            "confirmed_client_auc",
            "confirmed_client_precision",
            "confirmed_client_recall",
            "confirmed_client_f1",
        ]

        self.node_handle = self.node_path.open("w", newline="", encoding="utf-8")
        self.client_handle = self.client_path.open("w", newline="", encoding="utf-8")
        self.summary_handle = self.summary_path.open("w", newline="", encoding="utf-8")
        self.node_writer = csv.DictWriter(self.node_handle, fieldnames=self.node_fields)
        self.client_writer = csv.DictWriter(self.client_handle, fieldnames=self.client_fields)
        self.summary_writer = csv.DictWriter(self.summary_handle, fieldnames=self.summary_fields)
        self.node_writer.writeheader()
        self.client_writer.writeheader()
        self.summary_writer.writeheader()
        self.node_handle.flush()
        self.client_handle.flush()
        self.summary_handle.flush()

    def update_from_local_results(self, round_idx: int, local_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
        node_rows, client_rows, _ = compute_client_risk_features_from_local_results(
            local_results=local_results,
            malicious_client_ids=sorted(self.malicious_client_ids),
            highrisk_quantile=self.highrisk_quantile,
            suspicious_candidate_fraction=self.suspicious_candidate_fraction,
            benign_candidate_fraction=self.benign_candidate_fraction,
            highrisk_history_by_client_id=self.highrisk_history_by_client_id,
            early_history_by_client_id=self.early_history_by_client_id,
            early_threshold_mad_mult=self.early_threshold_mad_mult,
            confirmed_threshold_mad_mult=self.confirmed_threshold_mad_mult,
            confirmed_window=self.confirmed_window,
            confirmed_patience=self.confirmed_patience,
        )
        if not client_rows:
            return {}

        for row in node_rows:
            self.node_writer.writerow({"round": int(round_idx), **row})
        self.node_handle.flush()

        ranked_rows = sorted(client_rows, key=lambda item: float(item["risk_score"]), reverse=True)
        for row in client_rows:
            writer_row = {"round": int(round_idx), **{field: row.get(field, "") for field in self.client_fields if field != "round"}}
            self.client_writer.writerow(writer_row)
        self.client_handle.flush()

        summary_row = self._build_round_summary(round_idx=round_idx, client_rows=client_rows, ranked_rows=ranked_rows)
        self.summary_writer.writerow(summary_row)
        self.summary_handle.flush()
        self.round_summary_rows.append(dict(summary_row))
        self.latest_node_rows = [dict(row) for row in node_rows]
        self.latest_client_rows = [dict(row) for row in client_rows]
        self.latest_summary_row = dict(summary_row)
        return summary_row

    def _build_round_summary(
        self,
        round_idx: int,
        client_rows: Sequence[Dict[str, object]],
        ranked_rows: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        y_true = np.asarray([int(row["is_malicious"]) for row in client_rows], dtype=np.int64)
        y_score = np.asarray([float(row["risk_score"]) for row in client_rows], dtype=np.float64)
        y_pred_early = np.asarray([int(row["early_flag"]) for row in client_rows], dtype=np.int64)
        y_pred_confirmed = np.asarray([int(row["confirmed_flag"]) for row in client_rows], dtype=np.int64)

        malicious_scores = y_score[y_true == 1]
        benign_scores = y_score[y_true == 0]
        mean_malicious_risk = float(np.mean(malicious_scores)) if malicious_scores.size > 0 else float("nan")
        mean_benign_risk = float(np.mean(benign_scores)) if benign_scores.size > 0 else float("nan")
        malicious_risk_gap = (
            mean_malicious_risk - mean_benign_risk
            if malicious_scores.size > 0 and benign_scores.size > 0
            else float("nan")
        )

        topk = max(1, len(self.malicious_client_ids)) if self.malicious_client_ids else max(1, len(client_rows) // 3)
        topk_rows = list(ranked_rows[:topk])
        topk_ids = [int(row["client_id"]) for row in topk_rows]
        topk_hit_count = int(sum(int(client_id in self.malicious_client_ids) for client_id in topk_ids))
        topk_recall = (
            float(topk_hit_count / max(1, len(self.malicious_client_ids)))
            if self.malicious_client_ids
            else float("nan")
        )

        malicious_ranks = [int(row["risk_rank"]) for row in client_rows if int(row["is_malicious"]) == 1]
        avg_malicious_rank = float(np.mean(malicious_ranks)) if malicious_ranks else float("nan")
        best_malicious_rank = int(min(malicious_ranks)) if malicious_ranks else -1
        worst_malicious_rank = int(max(malicious_ranks)) if malicious_ranks else -1

        summary = {
            "round": int(round_idx),
            "mean_malicious_risk": mean_malicious_risk,
            "mean_benign_risk": mean_benign_risk,
            "malicious_risk_gap": malicious_risk_gap,
            "topk_client_ids": "|".join(str(client_id) for client_id in topk_ids),
            "topk_hit_count": int(topk_hit_count),
            "topk_recall": topk_recall,
            "avg_malicious_rank": avg_malicious_rank,
            "best_malicious_rank": best_malicious_rank,
            "worst_malicious_rank": worst_malicious_rank,
        }
        summary.update(summarize_detection(y_true, y_score, y_pred_early, prefix="early"))
        summary.update(summarize_detection(y_true, y_score, y_pred_confirmed, prefix="confirmed"))
        return summary


    def get_latest_client_rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self.latest_client_rows]

    def get_latest_node_rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self.latest_node_rows]

    def get_latest_client_feature_dict(self) -> Dict[int, Dict[str, float]]:
        feature_names = [
            "highrisk_level",
            "highrisk_growth",
            "consensus_margin",
            "label_alignment",
            "risk_score",
        ]
        payload: Dict[int, Dict[str, float]] = {}
        for row in self.latest_client_rows:
            payload[int(row["client_id"])] = {
                feature_name: float(row.get(feature_name, 0.0))
                for feature_name in feature_names
            }
        return payload

    def get_latest_summary_row(self) -> Dict[str, object]:
        return dict(self.latest_summary_row)

    def finalize(self) -> Dict[str, object]:
        self.node_handle.close()
        self.client_handle.close()
        self.summary_handle.close()
        if not self.round_summary_rows:
            return {}

        best_topk_row = max(
            self.round_summary_rows,
            key=lambda item: (float(item["topk_recall"]), float(item["confirmed_client_f1"])),
        )
        best_confirmed_row = max(
            self.round_summary_rows,
            key=lambda item: (float(item["confirmed_client_f1"]), float(item["topk_recall"])),
        )
        final_row = self.round_summary_rows[-1]
        return {
            "num_rounds": int(len(self.round_summary_rows)),
            "malicious_client_ids": sorted(int(client_id) for client_id in self.malicious_client_ids),
            "best_topk_recall_round": int(best_topk_row["round"]),
            "best_topk_recall": float(best_topk_row["topk_recall"]),
            "best_confirmed_f1_round": int(best_confirmed_row["round"]),
            "best_confirmed_f1": float(best_confirmed_row["confirmed_client_f1"]),
            "final_round": int(final_row["round"]),
            "final_topk_recall": float(final_row["topk_recall"]),
            "final_confirmed_f1": float(final_row["confirmed_client_f1"]),
            "final_mean_malicious_risk": float(final_row["mean_malicious_risk"]),
            "final_mean_benign_risk": float(final_row["mean_benign_risk"]),
        }
