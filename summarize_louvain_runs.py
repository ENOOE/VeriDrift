from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_pipe_ids(value: object) -> List[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    return [int(item) for item in text.split('|') if item.strip()]


def client_rows_to_dict(frame: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, row in frame.sort_values('client_id').iterrows():
        rows.append(
            {
                'client_id': int(row['client_id']),
                'risk_score': float(row['risk_score']),
                'risk_rank': int(row['risk_rank']),
                'early_flag': int(row['early_flag']),
                'confirmed_flag': int(row['confirmed_flag']),
                'highrisk_level': float(row['highrisk_level']),
                'highrisk_growth': float(row['highrisk_growth']),
                'consensus_margin': float(row['consensus_margin']),
                'label_alignment': float(row['label_alignment']),
            }
        )
    return rows


def cert_rows_to_dict(frame: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, row in frame.sort_values('client_id').iterrows():
        rows.append(
            {
                'client_id': int(row['client_id']),
                'risk_score': float(row['risk_score']),
                'risk_pressure': float(row['risk_pressure']),
                'certification_score': float(row['certification_score']),
                'certified_risk': float(row['certified_risk']),
                'aggregation_weight_ratio': float(row['aggregation_weight_ratio']),
                'cos_to_risk_weighted_center': float(row['cos_to_risk_weighted_center']),
                'update_anomaly': float(row['update_anomaly']),
                'monotonic_residual': float(row['monotonic_residual']),
            }
        )
    return rows


def summarise_run(run_dir: Path) -> Dict[str, object]:
    metrics = pd.read_csv(run_dir / 'metrics.csv')
    metrics['asr'] = pd.to_numeric(metrics['asr'], errors='coerce')
    attack_metadata = json.loads((run_dir / 'attack_metadata.json').read_text(encoding='utf-8'))
    risk_credit = pd.read_csv(run_dir / 'risk_credit' / 'client_risk_by_round.csv')
    risk_summary = pd.read_csv(run_dir / 'risk_credit' / 'round_detection_summary.csv')
    cert_credit = pd.read_csv(run_dir / 'server_certifier' / 'client_certification_by_round.csv')
    cert_summary = pd.read_csv(run_dir / 'server_certifier' / 'round_certification_summary.csv')
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    final_metrics = json.loads((run_dir / 'final_metrics.json').read_text(encoding='utf-8'))

    last_round = int(metrics['round'].max())
    last5_metrics = metrics.sort_values('round').tail(5).copy()
    last5_metrics_valid_asr = last5_metrics.dropna(subset=['asr']).copy()

    best_acc_row = last5_metrics.sort_values(['test_accuracy', 'asr'], ascending=[False, True]).iloc[0]
    if last5_metrics_valid_asr.empty:
        best_asr_row = last5_metrics.sort_values(['test_accuracy'], ascending=[False]).iloc[0]
    else:
        best_asr_row = last5_metrics_valid_asr.sort_values(['asr', 'test_accuracy'], ascending=[True, False]).iloc[0]

    final_risk_rows = risk_credit[risk_credit['round'] == last_round].copy()
    final_cert_rows = cert_credit[cert_credit['round'] == last_round].copy()
    last5_cert_rows = cert_credit[cert_credit['round'].isin(last5_metrics['round'])].copy()

    avg_weight_ratio_last5 = (
        last5_cert_rows.groupby('client_id', as_index=False)['aggregation_weight_ratio']
        .mean()
        .sort_values('client_id')
    )
    avg_weight_ratio_last5['aggregation_weight_ratio'] = avg_weight_ratio_last5['aggregation_weight_ratio'].astype(float)

    final_risk_summary = risk_summary[risk_summary['round'] == last_round].iloc[0]
    final_cert_summary = cert_summary[cert_summary['round'] == last_round].iloc[0]

    final_topk_risk_ids = parse_pipe_ids(final_risk_summary['topk_client_ids'])
    final_top_risk_ids = parse_pipe_ids(final_cert_summary['top_risk_client_ids'])
    final_top_down_ids = parse_pipe_ids(final_cert_summary['top_weighted_down_client_ids'])
    confirmed_detected_ids = [
        int(client_id)
        for client_id in final_risk_rows.loc[final_risk_rows['confirmed_flag'] == 1, 'client_id'].tolist()
    ]
    early_detected_ids = [
        int(client_id)
        for client_id in final_risk_rows.loc[final_risk_rows['early_flag'] == 1, 'client_id'].tolist()
    ]

    summary = {
        'run_dir': str(run_dir),
        'dataset': str(config['dataset']['name']),
        'malicious_client_ids': [int(client_id) for client_id in attack_metadata['malicious_client_ids']],
        'final_round': last_round,
        'final_metrics': {
            'selected_round': int(final_metrics['selected_round']),
            'val_accuracy': float(final_metrics['val_accuracy']),
            'test_accuracy': float(final_metrics['test_accuracy']),
            'clean_test_accuracy': float(final_metrics.get('clean_test_accuracy', float('nan'))),
            'asr': float(final_metrics['asr']) if str(final_metrics['asr']) != '' else float('nan'),
        },
        'last5_average': {
            'test_accuracy': float(last5_metrics['test_accuracy'].mean()),
            'asr': float(last5_metrics_valid_asr['asr'].mean()) if not last5_metrics_valid_asr.empty else float('nan'),
        },
        'last5_best_accuracy_round': {
            'round': int(best_acc_row['round']),
            'test_accuracy': float(best_acc_row['test_accuracy']),
            'asr': float(best_acc_row['asr']) if not pd.isna(best_acc_row['asr']) else float('nan'),
        },
        'last5_lowest_asr_round': {
            'round': int(best_asr_row['round']),
            'test_accuracy': float(best_asr_row['test_accuracy']),
            'asr': float(best_asr_row['asr']) if not pd.isna(best_asr_row['asr']) else float('nan'),
        },
        'risk_detection': {
            'final_topk_risk_client_ids': final_topk_risk_ids,
            'final_early_flag_client_ids': early_detected_ids,
            'final_confirmed_flag_client_ids': confirmed_detected_ids,
            'final_topk_hit_count': int(final_risk_summary['topk_hit_count']),
            'final_confirmed_f1': float(final_risk_summary['confirmed_client_f1']),
        },
        'continuous_certification': {
            'final_top_risk_client_ids': final_top_risk_ids,
            'final_top_weighted_down_client_ids': final_top_down_ids,
            'final_malicious_weight_mass': float(final_cert_summary['malicious_weight_mass']),
            'final_benign_weight_mass': float(final_cert_summary['benign_weight_mass']),
            'final_effective_risk_alpha': float(final_cert_summary['effective_risk_alpha']),
            'final_mean_malicious_certified_risk': float(final_cert_summary['mean_malicious_certified_risk']),
            'final_mean_benign_certified_risk': float(final_cert_summary['mean_benign_certified_risk']),
        },
        'final_round_risk_clients': client_rows_to_dict(final_risk_rows),
        'final_round_certification_clients': cert_rows_to_dict(final_cert_rows),
        'last5_avg_weight_ratio_by_client': [
            {
                'client_id': int(row['client_id']),
                'aggregation_weight_ratio': float(row['aggregation_weight_ratio']),
            }
            for _, row in avg_weight_ratio_last5.iterrows()
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize Louvain runs for node_case_risk_from_local.')
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    payload = {'runs': [summarise_run(run_dir.resolve()) for run_dir in args.run_dir]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
