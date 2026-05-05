from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


def flatten_floating_update(global_state_dict: Dict[str, torch.Tensor], local_state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for key, global_value in global_state_dict.items():
        if not torch.is_floating_point(global_value):
            continue
        delta = (local_state_dict[key].float() - global_value.detach().cpu().float()).reshape(-1).cpu().numpy()
        chunks.append(delta)
    if not chunks:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64, copy=False)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))


def sign_agreement_ratio(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    active = (np.abs(vec_a) > 1e-10) | (np.abs(vec_b) > 1e-10)
    if not np.any(active):
        return 1.0
    return float(np.mean(np.sign(vec_a[active]) == np.sign(vec_b[active])))


def robust_zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad <= 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - median) / (1.4826 * mad), -5.0, 5.0)


def softplus(values: np.ndarray, sharpness: float = 2.0) -> np.ndarray:
    scaled = np.clip(float(sharpness) * values, -40.0, 40.0)
    return np.log1p(np.exp(scaled)) / float(sharpness)


def robust_positive_scale(values: Sequence[float]) -> np.ndarray:
    z = robust_zscore(values)
    pressure = softplus(z)
    pressure = pressure - float(np.min(pressure))
    return pressure.astype(np.float64, copy=False)


def leave_one_out_update_center(
    update_dirs: Sequence[np.ndarray],
    previous_verified_risks: Sequence[float],
    target_idx: int,
) -> np.ndarray:
    other_indices = [idx for idx in range(len(update_dirs)) if idx != int(target_idx)]
    if not other_indices:
        return np.zeros_like(update_dirs[int(target_idx)], dtype=np.float64)

    def normalized_average(indices: Sequence[int]) -> np.ndarray:
        average = np.mean([update_dirs[idx] for idx in indices], axis=0)
        norm = float(np.linalg.norm(average))
        return np.zeros_like(average, dtype=np.float64) if norm <= 1e-12 else average / norm

    if len(other_indices) == 1:
        return normalized_average(other_indices)

    weights = np.exp(-np.asarray([previous_verified_risks[idx] for idx in other_indices], dtype=np.float64))
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return normalized_average(other_indices)

    center = np.zeros_like(update_dirs[int(target_idx)], dtype=np.float64)
    for weight, idx in zip(weights, other_indices):
        center += float(weight / weight_sum) * update_dirs[idx].astype(np.float64, copy=False)
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 1e-12:
        return normalized_average(other_indices)
    return center / center_norm


def leave_one_out_isotonic_predictions(risk_pressure: np.ndarray, update_anomaly: np.ndarray) -> np.ndarray:
    risk_pressure = np.asarray(risk_pressure, dtype=np.float64)
    update_anomaly = np.asarray(update_anomaly, dtype=np.float64)
    predictions = np.zeros_like(update_anomaly, dtype=np.float64)
    if len(risk_pressure) < 3 or float(np.max(risk_pressure) - np.min(risk_pressure)) <= 1e-12:
        predictions.fill(float(np.mean(update_anomaly)))
        return predictions

    for idx in range(len(risk_pressure)):
        mask = np.ones(len(risk_pressure), dtype=bool)
        mask[idx] = False
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(risk_pressure[mask], update_anomaly[mask])
        predictions[idx] = float(model.predict([risk_pressure[idx]])[0])
    return predictions


class ProxySupervisedTrustCertifier:
    """Continuous risk-score consistency certifier.

    The historical class name is kept so the training script can reuse the
    existing interface. This implementation does not train a verifier, does not
    use pseudo labels, and does not split clients into trusted/suspicious/reject
    groups. The main detector's risk score remains the primary down-weighting
    signal; update geometry only certifies whether that risk score is coherent.
    """

    def __init__(
        self,
        output_dir: Path,
        malicious_client_ids: Sequence[int] | None = None,
        device: str = "cpu",
        risk_alpha: float = 10.0,
        consistency_bonus: float = 0.5,
        warmup_tau: float = 20.0,
        risk_memory_decay: float = 0.90,
        **_: object,
    ) -> None:
        del device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.malicious_client_ids = {int(client_id) for client_id in (malicious_client_ids or [])}
        self.risk_alpha = float(risk_alpha)
        self.consistency_bonus = float(consistency_bonus)
        self.warmup_tau = float(warmup_tau)
        self.risk_memory_decay = float(risk_memory_decay)
        self.verified_risk_memory_by_client: Dict[int, float] = {}
        self.latest_weights: List[float] = []
        self.latest_round_rows: List[Dict[str, object]] = []
        self.round_summary_rows: List[Dict[str, object]] = []

        self.client_path = self.output_dir / "client_certification_by_round.csv"
        self.summary_path = self.output_dir / "round_certification_summary.csv"
        self.client_handle = self.client_path.open("w", newline="", encoding="utf-8")
        self.summary_handle = self.summary_path.open("w", newline="", encoding="utf-8")
        self.client_fields = [
            "round",
            "stage",
            "client_id",
            "is_malicious",
            "num_samples",
            "risk_score",
            "current_risk_pressure",
            "risk_pressure",
            "early_flag",
            "confirmed_flag",
            "update_norm",
            "update_norm_rz",
            "cos_to_risk_weighted_center",
            "sign_agreement",
            "update_anomaly",
            "isotonic_update_anomaly",
            "monotonic_residual",
            "residual_rz",
            "certification_score",
            "certified_risk",
            "aggregation_weight",
            "aggregation_weight_ratio",
        ]
        self.summary_fields = [
            "round",
            "stage",
            "num_clients",
            "risk_alpha",
            "effective_risk_alpha",
            "consistency_bonus",
            "warmup_tau",
            "risk_memory_decay",
            "mean_malicious_risk_pressure",
            "mean_benign_risk_pressure",
            "mean_malicious_certified_risk",
            "mean_benign_certified_risk",
            "mean_malicious_weight_ratio",
            "mean_benign_weight_ratio",
            "malicious_weight_mass",
            "benign_weight_mass",
            "mean_certification_score",
            "mean_monotonic_residual",
            "top_weighted_down_client_ids",
            "top_risk_client_ids",
            "top_risk_hit_count",
            "num_trusted",
            "num_suspicious",
            "num_reject",
            "rejected_client_ratio",
            "accepted_suspicious_weight_ratio",
            "mean_suspicious_trust_probability",
            "mlp_trained",
            "mlp_loss",
        ]
        self.client_writer = csv.DictWriter(self.client_handle, fieldnames=self.client_fields)
        self.summary_writer = csv.DictWriter(self.summary_handle, fieldnames=self.summary_fields)
        self.client_writer.writeheader()
        self.summary_writer.writeheader()
        self.client_handle.flush()
        self.summary_handle.flush()

    def pretrain(self, *_: object, **__: object) -> Dict[str, object]:
        return {}

    def _prepare_rows(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        local_results: Sequence[Dict[str, object]],
        client_risk_rows: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        risk_by_client = {int(row["client_id"]): dict(row) for row in client_risk_rows}
        rows: List[Dict[str, object]] = []
        for local_result in local_results:
            client_id = int(local_result["risk_payload"]["client_id"])
            risk_row = risk_by_client.get(client_id, {})
            update_vector = flatten_floating_update(global_state_dict, local_result["state_dict"])
            update_norm = float(np.linalg.norm(update_vector))
            rows.append(
                {
                    "client_id": client_id,
                    "is_malicious": int(client_id in self.malicious_client_ids),
                    "num_samples": int(local_result["num_samples"]),
                    "risk_score": float(risk_row.get("risk_score", 0.0)),
                    "early_flag": int(float(risk_row.get("early_flag", 0.0)) >= 0.5),
                    "confirmed_flag": int(float(risk_row.get("confirmed_flag", 0.0)) >= 0.5),
                    "update_vector": update_vector,
                    "update_norm": float(update_norm),
                    "update_dir": update_vector / (update_norm + 1e-12),
                    "previous_verified_risk": float(self.verified_risk_memory_by_client.get(client_id, 0.0)),
                }
            )

        current_risk_pressure = robust_positive_scale([row["risk_score"] for row in rows])
        for idx, row in enumerate(rows):
            current_pressure = float(current_risk_pressure[idx])
            row["current_risk_pressure"] = float(current_pressure)
            row["risk_pressure"] = float(current_pressure)
        risk_pressure = np.asarray([row["risk_pressure"] for row in rows], dtype=np.float64)
        update_norm_z = robust_zscore([row["update_norm"] for row in rows])
        update_dirs = [row["update_dir"] for row in rows]
        previous_verified_risks = [float(row["previous_verified_risk"]) for row in rows]

        anomaly_values: List[float] = []
        for idx, row in enumerate(rows):
            center = leave_one_out_update_center(
                update_dirs=update_dirs,
                previous_verified_risks=previous_verified_risks,
                target_idx=idx,
            )
            cos_to_center = cosine_similarity(row["update_dir"], center)
            update_anomaly = (1.0 - cos_to_center) + max(0.0, float(update_norm_z[idx]))
            row["update_norm_rz"] = float(update_norm_z[idx])
            row["cos_to_risk_weighted_center"] = float(cos_to_center)
            row["sign_agreement"] = float(sign_agreement_ratio(row["update_dir"], center))
            row["update_anomaly"] = float(update_anomaly)
            anomaly_values.append(float(update_anomaly))

        isotonic_pred = leave_one_out_isotonic_predictions(risk_pressure, np.asarray(anomaly_values, dtype=np.float64))
        residuals = np.abs(np.asarray(anomaly_values, dtype=np.float64) - isotonic_pred)
        residual_rz = np.maximum(0.0, robust_zscore(residuals))
        certification_scores = np.exp(-residual_rz)

        for idx, row in enumerate(rows):
            risk_pressure_value = float(row["risk_pressure"])
            certification_score = float(certification_scores[idx])
            certified_risk = (
                risk_pressure_value * (1.0 + self.consistency_bonus * certification_score)
                + self.consistency_bonus
                * (1.0 - certification_score)
                / (1.0 + risk_pressure_value)
            )
            row["isotonic_update_anomaly"] = float(isotonic_pred[idx])
            row["monotonic_residual"] = float(residuals[idx])
            row["residual_rz"] = float(residual_rz[idx])
            row["certification_score"] = float(certification_score)
            row["certified_risk"] = float(certified_risk)
        for row in rows:
            client_id = int(row["client_id"])
            previous_verified_risk = self.verified_risk_memory_by_client.get(client_id, 0.0)
            self.verified_risk_memory_by_client[client_id] = (
                self.risk_memory_decay * float(previous_verified_risk)
                + (1.0 - self.risk_memory_decay) * float(row["certified_risk"])
            )
        return rows

    def certify_round(
        self,
        round_idx: int,
        global_model,
        global_state_dict: Dict[str, torch.Tensor],
        local_results: Sequence[Dict[str, object]],
        client_risk_rows: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        del global_model
        stage = "continuous"
        rows = self._prepare_rows(
            global_state_dict=global_state_dict,
            local_results=local_results,
            client_risk_rows=client_risk_rows,
        )

        effective_risk_alpha = self.risk_alpha * (1.0 - float(np.exp(-float(round_idx) / max(1e-8, self.warmup_tau))))
        weights = [
            float(row["num_samples"]) * float(np.exp(-effective_risk_alpha * float(row["certified_risk"])))
            for row in rows
        ]
        total_weight = float(sum(weights))
        if total_weight <= 1e-12:
            weights = [float(max(1, int(row["num_samples"]))) for row in rows]
            total_weight = float(sum(weights))

        for row, weight in zip(rows, weights):
            row["aggregation_weight"] = float(weight)
            row["aggregation_weight_ratio"] = float(weight / total_weight)
            self.client_writer.writerow(
                {
                    field: (
                        int(round_idx)
                        if field == "round"
                        else stage
                        if field == "stage"
                        else row.get(field, "")
                    )
                    for field in self.client_fields
                }
            )
        self.client_handle.flush()

        malicious_rows = [row for row in rows if int(row["is_malicious"]) == 1]
        benign_rows = [row for row in rows if int(row["is_malicious"]) == 0]
        top_count = max(1, len(self.malicious_client_ids)) if self.malicious_client_ids else max(1, len(rows) // 3)
        top_down_rows = sorted(rows, key=lambda row: float(row["aggregation_weight_ratio"]))[:top_count]
        top_risk_rows = sorted(rows, key=lambda row: float(row["risk_score"]), reverse=True)[:top_count]
        top_risk_ids = [int(row["client_id"]) for row in top_risk_rows]
        top_risk_hit_count = int(sum(client_id in self.malicious_client_ids for client_id in top_risk_ids))

        summary = {
            "round": int(round_idx),
            "stage": stage,
            "num_clients": int(len(rows)),
            "risk_alpha": float(self.risk_alpha),
            "effective_risk_alpha": float(effective_risk_alpha),
            "consistency_bonus": float(self.consistency_bonus),
            "warmup_tau": float(self.warmup_tau),
            "risk_memory_decay": float(self.risk_memory_decay),
            "mean_malicious_risk_pressure": float(np.mean([row["risk_pressure"] for row in malicious_rows])) if malicious_rows else float("nan"),
            "mean_benign_risk_pressure": float(np.mean([row["risk_pressure"] for row in benign_rows])) if benign_rows else float("nan"),
            "mean_malicious_certified_risk": float(np.mean([row["certified_risk"] for row in malicious_rows])) if malicious_rows else float("nan"),
            "mean_benign_certified_risk": float(np.mean([row["certified_risk"] for row in benign_rows])) if benign_rows else float("nan"),
            "mean_malicious_weight_ratio": float(np.mean([row["aggregation_weight_ratio"] for row in malicious_rows])) if malicious_rows else float("nan"),
            "mean_benign_weight_ratio": float(np.mean([row["aggregation_weight_ratio"] for row in benign_rows])) if benign_rows else float("nan"),
            "malicious_weight_mass": float(sum(row["aggregation_weight_ratio"] for row in malicious_rows)),
            "benign_weight_mass": float(sum(row["aggregation_weight_ratio"] for row in benign_rows)),
            "mean_certification_score": float(np.mean([row["certification_score"] for row in rows])),
            "mean_monotonic_residual": float(np.mean([row["monotonic_residual"] for row in rows])),
            "top_weighted_down_client_ids": "|".join(str(int(row["client_id"])) for row in top_down_rows),
            "top_risk_client_ids": "|".join(str(client_id) for client_id in top_risk_ids),
            "top_risk_hit_count": int(top_risk_hit_count),
            "num_trusted": int(len(rows)),
            "num_suspicious": 0,
            "num_reject": 0,
            "rejected_client_ratio": 0.0,
            "accepted_suspicious_weight_ratio": 0.0,
            "mean_suspicious_trust_probability": float(np.mean([row["certification_score"] for row in rows])),
            "mlp_trained": 0.0,
            "mlp_loss": float("nan"),
        }
        self.summary_writer.writerow(summary)
        self.summary_handle.flush()
        self.latest_weights = [float(row["aggregation_weight"]) for row in rows]
        self.latest_round_rows = [dict(row) for row in rows]
        self.round_summary_rows.append(dict(summary))
        return dict(summary)

    def get_latest_weights(self) -> List[float]:
        return list(self.latest_weights)

    def get_latest_round_rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self.latest_round_rows]

    def finalize(self) -> Dict[str, object]:
        self.client_handle.close()
        self.summary_handle.close()
        if not self.round_summary_rows:
            return {
                "certifier": "risk_score_consistency",
                "risk_alpha": float(self.risk_alpha),
                "warmup_tau": float(self.warmup_tau),
                "risk_memory_decay": float(self.risk_memory_decay),
                "consistency_bonus": float(self.consistency_bonus),
            }
        final = self.round_summary_rows[-1]
        best_downweight_row = min(
            self.round_summary_rows,
            key=lambda row: float(row["malicious_weight_mass"]),
        )
        return {
            "certifier": "risk_score_consistency",
            "num_rounds": int(len(self.round_summary_rows)),
            "risk_alpha": float(self.risk_alpha),
            "warmup_tau": float(self.warmup_tau),
            "risk_memory_decay": float(self.risk_memory_decay),
            "consistency_bonus": float(self.consistency_bonus),
            "final_round": int(final["round"]),
            "final_stage": str(final["stage"]),
            "final_malicious_weight_mass": float(final["malicious_weight_mass"]),
            "final_benign_weight_mass": float(final["benign_weight_mass"]),
            "final_mean_malicious_certified_risk": float(final["mean_malicious_certified_risk"]),
            "final_mean_benign_certified_risk": float(final["mean_benign_certified_risk"]),
            "final_top_weighted_down_client_ids": str(final["top_weighted_down_client_ids"]),
            "final_top_risk_client_ids": str(final["top_risk_client_ids"]),
            "final_top_risk_hit_count": int(final["top_risk_hit_count"]),
            "best_downweight_round": int(best_downweight_row["round"]),
            "best_malicious_weight_mass": float(best_downweight_row["malicious_weight_mass"]),
        }
