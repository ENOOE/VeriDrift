from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return np.zeros_like(arr)
    return np.clip((arr - median) / (1.4826 * mad), -5.0, 5.0)


class TrustMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ProxySupervisedTrustCertifier:
    """Legacy risk-credit certifier kept under the old import name.

    This compatibility implementation keeps the observe, calibration, and active
    stages used by earlier risk-credit experiments.
    """

    def __init__(
        self,
        output_dir: Path,
        malicious_client_ids: Sequence[int] | None = None,
        device: str = "cpu",
        observe_rounds: int = 10,
        calibration_rounds: int = 10,
        trusted_risk_quantile: float = 0.4,
        trusted_cos_quantile: float = 0.6,
        suspicious_weight_floor: float = 0.5,
        **_: object,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.malicious_client_ids = {int(client_id) for client_id in (malicious_client_ids or [])}
        self.device = torch.device(device)
        self.observe_rounds = int(observe_rounds)
        self.calibration_rounds = int(calibration_rounds)
        self.active_start_round = self.observe_rounds + self.calibration_rounds + 1
        self.trusted_risk_quantile = float(trusted_risk_quantile)
        self.trusted_cos_quantile = float(trusted_cos_quantile)
        self.suspicious_weight_floor = float(suspicious_weight_floor)

        self.model = TrustMLP(input_dim=5).to(self.device)
        self.feature_center = np.zeros(5, dtype=np.float32)
        self.feature_scale = np.ones(5, dtype=np.float32)
        self.training_features: List[np.ndarray] = []
        self.training_labels: List[float] = []
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
            "early_flag",
            "confirmed_flag",
            "update_norm",
            "update_norm_rz",
            "cos_to_median_update",
            "cos_to_trusted_prototype",
            "sign_agreement",
            "trusted_risk_threshold",
            "trusted_cos_threshold",
            "rule_decision",
            "decision",
            "trust_probability",
            "trusted_confidence",
            "suspicious_confidence",
            "aggregation_weight",
            "aggregation_weight_ratio",
        ]
        self.summary_fields = [
            "round",
            "stage",
            "num_trusted",
            "num_suspicious",
            "num_reject",
            "rejected_client_ratio",
            "accepted_suspicious_weight_ratio",
            "trusted_weight_mass",
            "suspicious_weight_mass",
            "reject_weight_mass",
            "malicious_trusted",
            "malicious_suspicious",
            "malicious_reject",
            "benign_trusted",
            "benign_suspicious",
            "benign_reject",
            "trusted_risk_threshold",
            "trusted_cos_threshold",
            "mean_suspicious_trust_probability",
            "mlp_trained",
            "mlp_loss",
            "mlp_num_pos",
            "mlp_num_neg",
        ]
        self.client_writer = csv.DictWriter(self.client_handle, fieldnames=self.client_fields)
        self.summary_writer = csv.DictWriter(self.summary_handle, fieldnames=self.summary_fields)
        self.client_writer.writeheader()
        self.summary_writer.writeheader()
        self.client_handle.flush()
        self.summary_handle.flush()

    def pretrain(self, *_: object, **__: object) -> Dict[str, object]:
        return {}

    def _stage(self, round_idx: int) -> str:
        if round_idx <= self.observe_rounds:
            return "observe"
        if round_idx < self.active_start_round:
            return "calibration"
        return "active"

    def _feature_vector(self, row: Dict[str, object]) -> np.ndarray:
        return np.asarray(
            [
                float(row["risk_score"]),
                float(row["update_norm_rz"]),
                float(row["cos_to_median_update"]),
                float(row["cos_to_trusted_prototype"]),
                float(row["sign_agreement"]),
            ],
            dtype=np.float32,
        )

    def _fit_normalizer(self, features: np.ndarray) -> None:
        center = np.median(features, axis=0).astype(np.float32)
        mad = np.median(np.abs(features - center[None, :]), axis=0).astype(np.float32)
        self.feature_center = center
        self.feature_scale = np.maximum(1e-6, 1.4826 * mad).astype(np.float32)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        return np.clip((features.astype(np.float32) - self.feature_center[None, :]) / self.feature_scale[None, :], -8.0, 8.0)

    def _train_mlp(self) -> Tuple[float, int, int]:
        labels = np.asarray(self.training_labels, dtype=np.float32)
        if len(labels) < 8 or len(set(int(label) for label in labels)) < 2:
            return float("nan"), int(np.sum(labels >= 0.5)), int(np.sum(labels < 0.5))
        features = np.stack(self.training_features, axis=0).astype(np.float32)
        self._fit_normalizer(features)
        x = torch.tensor(self._normalize(features), dtype=torch.float32, device=self.device)
        y = torch.tensor(labels, dtype=torch.float32, device=self.device)
        pos = float(torch.sum(y >= 0.5).item())
        neg = float(torch.sum(y < 0.5).item())
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-4)
        losses: List[float] = []
        self.model.train()
        for _ in range(60):
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses[-10:])), int(pos), int(neg)

    def _predict_trust(self, row: Dict[str, object]) -> float:
        if len(self.training_labels) < 8 or len(set(int(label) for label in self.training_labels)) < 2:
            return 1.0
        with torch.no_grad():
            self.model.eval()
            features = self._feature_vector(row)[None, :]
            x = torch.tensor(self._normalize(features), dtype=torch.float32, device=self.device)
            return float(torch.sigmoid(self.model(x)).detach().cpu().item())

    def _prepare_rows(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        local_results: Sequence[Dict[str, object]],
        client_risk_rows: Sequence[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], float, float]:
        risk_by_client = {int(row["client_id"]): dict(row) for row in client_risk_rows}
        rows: List[Dict[str, object]] = []
        for local_result in local_results:
            client_id = int(local_result["risk_payload"]["client_id"])
            risk_row = risk_by_client.get(client_id, {})
            update_vector = flatten_floating_update(global_state_dict, local_result["state_dict"])
            rows.append(
                {
                    "client_id": client_id,
                    "is_malicious": int(client_id in self.malicious_client_ids),
                    "num_samples": int(local_result["num_samples"]),
                    "risk_score": float(risk_row.get("risk_score", 0.0)),
                    "early_flag": int(float(risk_row.get("early_flag", 0.0)) >= 0.5),
                    "confirmed_flag": int(float(risk_row.get("confirmed_flag", 0.0)) >= 0.5),
                    "update_vector": update_vector,
                    "update_norm": float(np.linalg.norm(update_vector)),
                }
            )

        update_norm_z = robust_zscore([row["update_norm"] for row in rows])
        median_update = np.median(np.stack([row["update_vector"] for row in rows], axis=0), axis=0)
        low_risk_count = max(1, int(round(self.trusted_risk_quantile * len(rows))))
        low_risk_rows = sorted(rows, key=lambda row: float(row["risk_score"]))[:low_risk_count]
        trusted_proto = np.mean(np.stack([row["update_vector"] for row in low_risk_rows], axis=0), axis=0)
        for idx, row in enumerate(rows):
            row["update_norm_rz"] = float(update_norm_z[idx])
            row["cos_to_median_update"] = cosine_similarity(row["update_vector"], median_update)
            row["cos_to_trusted_prototype"] = cosine_similarity(row["update_vector"], trusted_proto)
            row["sign_agreement"] = sign_agreement_ratio(row["update_vector"], trusted_proto)

        trusted_risk_threshold = float(np.quantile([row["risk_score"] for row in rows], self.trusted_risk_quantile))
        trusted_cos_threshold = float(np.quantile([row["cos_to_trusted_prototype"] for row in rows], self.trusted_cos_quantile))
        return rows, trusted_risk_threshold, trusted_cos_threshold

    def certify_round(
        self,
        round_idx: int,
        global_model,
        global_state_dict: Dict[str, torch.Tensor],
        local_results: Sequence[Dict[str, object]],
        client_risk_rows: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        del global_model
        stage = self._stage(round_idx)
        rows, trusted_risk_threshold, trusted_cos_threshold = self._prepare_rows(
            global_state_dict=global_state_dict,
            local_results=local_results,
            client_risk_rows=client_risk_rows,
        )

        for row in rows:
            rule_trusted = (
                float(row["risk_score"]) <= trusted_risk_threshold
                and float(row["cos_to_trusted_prototype"]) >= trusted_cos_threshold
            )
            rule_reject = bool(row["confirmed_flag"]) and float(row["risk_score"]) > trusted_risk_threshold
            if rule_trusted:
                rule_decision = "trusted"
            elif rule_reject:
                rule_decision = "reject"
            else:
                rule_decision = "suspicious"
            row["rule_decision"] = rule_decision

            if rule_decision == "trusted":
                self.training_features.append(self._feature_vector(row))
                self.training_labels.append(1.0)
            elif rule_decision == "reject" or (float(row["update_norm_rz"]) >= 4.0 and float(row["risk_score"]) > trusted_risk_threshold):
                self.training_features.append(self._feature_vector(row))
                self.training_labels.append(0.0)

        mlp_loss = float("nan")
        mlp_num_pos = int(sum(label >= 0.5 for label in self.training_labels))
        mlp_num_neg = int(sum(label < 0.5 for label in self.training_labels))
        mlp_trained = 0.0
        if stage != "observe":
            mlp_loss, mlp_num_pos, mlp_num_neg = self._train_mlp()
            mlp_trained = float(np.isfinite(mlp_loss))

        weights: List[float] = []
        suspicious_probs: List[float] = []
        for row in rows:
            rule_decision = str(row["rule_decision"])
            if stage == "observe":
                decision = "trusted" if rule_decision == "trusted" else "suspicious"
            elif stage == "calibration":
                decision = "trusted" if rule_decision == "trusted" else "suspicious"
            else:
                decision = rule_decision

            if decision == "trusted":
                trust_probability = 1.0
                weight = float(row["num_samples"])
            elif decision == "reject":
                trust_probability = 0.0
                weight = 0.0
            else:
                trust_probability = self._predict_trust(row)
                if stage != "active":
                    trust_probability = 1.0
                elif abs(trust_probability - 0.5) < 0.08:
                    trust_probability = self.suspicious_weight_floor
                suspicious_probs.append(float(trust_probability))
                weight = float(row["num_samples"]) * float(trust_probability)

            row["decision"] = decision
            row["trust_probability"] = float(trust_probability)
            row["trusted_confidence"] = float(row["cos_to_trusted_prototype"]) - trusted_cos_threshold
            row["suspicious_confidence"] = float(row["risk_score"]) - trusted_risk_threshold
            row["aggregation_weight"] = float(weight)
            weights.append(float(weight))

        total_weight = float(sum(weights))
        if total_weight <= 1e-12:
            weights = [float(max(1, int(row["num_samples"]))) for row in rows]
            total_weight = float(sum(weights))

        for row, weight in zip(rows, weights):
            row["aggregation_weight"] = float(weight)
            row["aggregation_weight_ratio"] = float(weight / total_weight)
            self.client_writer.writerow(
                {
                    "round": int(round_idx),
                    "stage": stage,
                    "client_id": int(row["client_id"]),
                    "is_malicious": int(row["is_malicious"]),
                    "num_samples": int(row["num_samples"]),
                    "risk_score": float(row["risk_score"]),
                    "early_flag": int(row["early_flag"]),
                    "confirmed_flag": int(row["confirmed_flag"]),
                    "update_norm": float(row["update_norm"]),
                    "update_norm_rz": float(row["update_norm_rz"]),
                    "cos_to_median_update": float(row["cos_to_median_update"]),
                    "cos_to_trusted_prototype": float(row["cos_to_trusted_prototype"]),
                    "sign_agreement": float(row["sign_agreement"]),
                    "trusted_risk_threshold": float(trusted_risk_threshold),
                    "trusted_cos_threshold": float(trusted_cos_threshold),
                    "rule_decision": str(row["rule_decision"]),
                    "decision": str(row["decision"]),
                    "trust_probability": float(row["trust_probability"]),
                    "trusted_confidence": float(row["trusted_confidence"]),
                    "suspicious_confidence": float(row["suspicious_confidence"]),
                    "aggregation_weight": float(row["aggregation_weight"]),
                    "aggregation_weight_ratio": float(row["aggregation_weight_ratio"]),
                }
            )
        self.client_handle.flush()

        def count(decision: str, malicious: int | None = None) -> int:
            return sum(
                int(row["decision"] == decision and (malicious is None or int(row["is_malicious"]) == malicious))
                for row in rows
            )

        trusted_weight_mass = float(sum(row["aggregation_weight"] for row in rows if row["decision"] == "trusted"))
        suspicious_weight_mass = float(sum(row["aggregation_weight"] for row in rows if row["decision"] == "suspicious"))
        reject_weight_mass = float(sum(row["num_samples"] for row in rows if row["decision"] == "reject"))
        summary = {
            "round": int(round_idx),
            "stage": stage,
            "num_trusted": count("trusted"),
            "num_suspicious": count("suspicious"),
            "num_reject": count("reject"),
            "rejected_client_ratio": float(count("reject") / max(1, len(rows))),
            "accepted_suspicious_weight_ratio": float(suspicious_weight_mass / total_weight),
            "trusted_weight_mass": trusted_weight_mass,
            "suspicious_weight_mass": suspicious_weight_mass,
            "reject_weight_mass": reject_weight_mass,
            "malicious_trusted": count("trusted", 1),
            "malicious_suspicious": count("suspicious", 1),
            "malicious_reject": count("reject", 1),
            "benign_trusted": count("trusted", 0),
            "benign_suspicious": count("suspicious", 0),
            "benign_reject": count("reject", 0),
            "trusted_risk_threshold": float(trusted_risk_threshold),
            "trusted_cos_threshold": float(trusted_cos_threshold),
            "mean_suspicious_trust_probability": float(np.mean(suspicious_probs)) if suspicious_probs else float("nan"),
            "mlp_trained": mlp_trained,
            "mlp_loss": float(mlp_loss),
            "mlp_num_pos": int(mlp_num_pos),
            "mlp_num_neg": int(mlp_num_neg),
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
                "observe_rounds": int(self.observe_rounds),
                "calibration_rounds": int(self.calibration_rounds),
                "active_start_round": int(self.active_start_round),
            }
        final = self.round_summary_rows[-1]
        best_round = max(
            self.round_summary_rows,
            key=lambda row: (int(row["malicious_reject"]), -int(row["benign_reject"])),
        )
        return {
            "num_rounds": int(len(self.round_summary_rows)),
            "observe_rounds": int(self.observe_rounds),
            "calibration_rounds": int(self.calibration_rounds),
            "active_start_round": int(self.active_start_round),
            "final_round": int(final["round"]),
            "final_stage": str(final["stage"]),
            "final_num_trusted": int(final["num_trusted"]),
            "final_num_suspicious": int(final["num_suspicious"]),
            "final_num_reject": int(final["num_reject"]),
            "final_rejected_client_ratio": float(final["rejected_client_ratio"]),
            "final_accepted_suspicious_weight_ratio": float(final["accepted_suspicious_weight_ratio"]),
            "final_mean_suspicious_trust_probability": float(final["mean_suspicious_trust_probability"]),
            "final_mlp_loss": float(final["mlp_loss"]),
            "final_mlp_num_pos": int(final["mlp_num_pos"]),
            "final_mlp_num_neg": int(final["mlp_num_neg"]),
            "best_malicious_reject_round": int(best_round["round"]),
            "best_malicious_reject_count": int(best_round["malicious_reject"]),
            "best_benign_reject_count_at_best_round": int(best_round["benign_reject"]),
        }
