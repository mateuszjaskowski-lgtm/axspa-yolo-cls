#!/usr/bin/env python3
"""Reproduce nominal reader agreement from the ordered, nonidentifying CSV.

Run from the repository root:
    python code/recompute_reader_agreement.py

Only NumPy is required. Each row contains one pair of initial ratings and the
consensus label. Row indices preserve the original analysis order; they are not
patient identifiers or links to image files. No rows are sorted or reweighted.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


COLUMNS = ["row_index", "cohort", "mp_label", "pg_label", "consensus_label"]
CLASS_NAMES = ["normal", "osteofity", "parasyndesmofity", "syndesmofity"]
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0


def load_ratings(path):
    """Validate the published schema and retain its physical row order."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Expected exactly these ordered columns: {COLUMNS}")
        records = list(reader)
    if not records:
        raise ValueError("The reader-rating CSV is empty.")
    for index, row in enumerate(records, start=1):
        if int(row["row_index"]) != index:
            raise ValueError("Row indices must be consecutive and preserve CSV order.")
        if row["cohort"] not in {"Lodz", "Konskie"}:
            raise ValueError("Unexpected cohort label.")
        for key in ["mp_label", "pg_label", "consensus_label"]:
            if row[key] not in {"0", "1", "2", "3"}:
                raise ValueError(f"Invalid nominal label in {key} at row {index}.")
    return records


def kappa(x, y):
    """Same four-class nominal-kappa calculation as the original audit."""
    matrix = np.bincount(x * 4 + y, minlength=16).reshape(4, 4)
    n = matrix.sum()
    observed = np.trace(matrix) / n
    expected = np.sum(matrix.sum(0) * matrix.sum(1)) / (n * n)
    return float((observed - expected) / (1 - expected)), matrix.tolist()


def calculate(records):
    results = {}
    for cohort in ["All", "Lodz", "Konskie"]:
        rows = records if cohort == "All" else [r for r in records if r["cohort"] == cohort]
        if not rows:
            raise ValueError(f"No rows for {cohort}.")
        x = np.array([int(r["mp_label"]) for r in rows])
        y = np.array([int(r["pg_label"]) for r in rows])
        estimate, matrix = kappa(x, y)
        # Deliberately reset the generator for the pooled and each site analysis.
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot = []
        for _ in range(N_BOOTSTRAP):
            indices = rng.integers(0, len(rows), len(rows))
            boot.append(kappa(x[indices], y[indices])[0])
        if not np.all(np.isfinite(boot)):
            raise ValueError("An undefined bootstrap kappa occurred; inspect the input.")
        results[cohort] = {
            "n": len(rows),
            "agreement": int(np.sum(x == y)),
            "disagreement": int(np.sum(x != y)),
            "observed_agreement": float(np.mean(x == y)),
            "nominal_kappa": estimate,
            "ci95": np.quantile(boot, [.025, .975], method="linear").tolist(),
            "matrix_MP_rows_PG_columns": matrix,
            "consensus_counts": {str(label): sum(int(r["consensus_label"]) == label for r in rows)
                                 for label in range(4)},
            "consensus_outside_reader_choices": sum(r["consensus_label"] not in [r["mp_label"], r["pg_label"]]
                                                    for r in rows),
            "bootstrap_valid_replicates": int(np.sum(np.isfinite(boot))),
        }
    return results


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/reader_ratings.csv")
    parser.add_argument("--output", type=Path, default=root / "analysis/reader_agreement_recomputed.json")
    args = parser.parse_args()
    records = load_ratings(args.input)
    result = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "class_order": CLASS_NAMES,
        "numpy_version": np.__version__,
        "method": {
            "agreement": "Initial M.P. and P.G. labels agree before consensus.",
            "kappa": "Unweighted Cohen kappa for four nominal classes.",
            "ci": "95% percentile paired-row bootstrap; 10000 draws; default_rng(0) reset for All, Lodz and Konskie; linear quantiles.",
            "row_order": "Physical CSV row order, retained from the accepted workbook; no sorting.",
            "unit": "One accepted case per row; row_index is a technical sequence, not a patient or image identifier.",
        },
        "statistics": calculate(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cohort, item in result["statistics"].items():
        print(f"{cohort}: n={item['n']}; agreement={item['agreement']}; kappa={item['nominal_kappa']:.12f}; CI={item['ci95']}")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
