"""Independently check public point estimates without PNGs, weights or torch."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

CLASSES = ['normal', 'osteofity', 'parasyndesmofity', 'syndesmofity']
MODELS = ['yolo11l-cls', 'yolo26l-cls', 'resnet50', 'efficientnet_b0']


def check_all(analysis):
    checked = 0
    seed_accuracies = {}
    mismatches = []
    for model in MODELS:
        seed_accuracies[model] = []
        for seed in range(5):
            for split, expected_n in [('val', 95), ('test', 150)]:
                folder = analysis / model / f'seed_{seed}'
                with (folder / f'{split}_predictions.csv').open(newline='') as f:
                    rows = list(csv.DictReader(f))
                if len(rows) != expected_n or len({r['image_id'] for r in rows}) != expected_n:
                    raise ValueError(f'Unexpected sample count or repeated ID: {folder}/{split}')
                y = np.array([int(r['true_label']) for r in rows])
                p = np.array([[float(r['p_' + c]) for c in CLASSES] for r in rows])
                if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
                    raise ValueError(f'Invalid probabilities: {folder}/{split}')
                pred = p.argmax(1)
                if not np.array_equal(pred, [int(r['predicted_label']) for r in rows]):
                    raise ValueError(f'Prediction/probability mismatch: {folder}/{split}')
                binary_y = (y >= 2).astype(int)
                binary_p = p[:, 2:].sum(1)
                brier = np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1))
                metrics = {
                    'accuracy': accuracy_score(y, pred),
                    'balanced_accuracy': balanced_accuracy_score(y, pred),
                    'unweighted_cohen_kappa': cohen_kappa_score(y, pred),
                    'macro_f1': f1_score(y, pred, labels=range(4), average='macro'),
                    'macro_ovr_auc': np.mean([roc_auc_score(y == c, p[:, c]) for c in range(4)]),
                    'binary_auc': roc_auc_score(binary_y, binary_p),
                    'multiclass_brier': brier,
                    'mean_ovr_brier': brier / 4,
                    'binary_brier': np.mean((binary_p - binary_y) ** 2),
                }
                stored = json.loads((folder / f'{split}_metrics.json').read_text())['metrics']
                for name, actual in metrics.items():
                    expected = stored[name]['estimate']
                    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
                        mismatches.append({'model': model, 'seed': seed, 'split': split, 'metric': name, 'computed': float(actual), 'stored': expected})
                    checked += 1
                if split == 'test':
                    seed_accuracies[model].append(float(metrics['accuracy']))
    result = {
        'status': 'PASS' if not mismatches else 'FAIL',
        'point_estimates_checked': checked,
        'prediction_tables_checked': 40,
        'scope': 'Point estimates only; confidence intervals and paired tests are preserved in the frozen evaluation outputs.',
        'external_accuracy_by_seed': seed_accuracies,
        'five_seed_accuracy': {m: {'mean': float(np.mean(v)), 'sample_sd': float(np.std(v, ddof=1))} for m, v in seed_accuracies.items()},
        'mismatches': mismatches,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=Path(__file__).resolve().parents[1] / 'analysis')
    args = parser.parse_args()
    result = check_all(args.analysis)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
