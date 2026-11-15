"""Descriptive reader agreement with a co-created consensus; not a human benchmark.

Uses corrected pre-consensus ratings on the Konskie external cohort only.
No model comparison, hypothesis test, or noninferiority analysis is performed.
The source CSV is read only; all outputs go into the requested analysis folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

LABELS = [0, 1, 2, 3]
CLASSES = ["normal", "osteophytes", "parasyndesmophytes", "syndesmophytes"]
INTERPRETATION = (
    "Descriptive agreement with consensus co-created by the same reader; "
    "dependent reference/incorporation bias, not an independent human benchmark. "
    "No comparison with model performance, p-values, or noninferiority claims."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> list[float]:
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return [center - half_width, center + half_width]


def calculate(rows: list[dict[str, str]], reader_field: str) -> dict:
    matrix = [[0] * 4 for _ in LABELS]
    for row in rows:
        consensus = int(row["consensus_label"])
        initial = int(row[reader_field])
        if consensus not in LABELS or initial not in LABELS:
            raise ValueError("Unknown class label")
        matrix[consensus][initial] += 1
    n = len(rows)
    correct = sum(matrix[i][i] for i in LABELS)
    reference_totals = [sum(row) for row in matrix]
    reader_totals = [sum(matrix[i][j] for i in LABELS) for j in LABELS]
    observed = correct / n
    expected = sum(a * b for a, b in zip(reference_totals, reader_totals)) / (n * n)
    kappa = (observed - expected) / (1 - expected)
    assert sum(map(sum, matrix)) == n == 150
    assert reference_totals == [55, 54, 20, 21]
    return {
        "n": n,
        "matches_consensus": correct,
        "differs_from_consensus": n - correct,
        "accuracy_against_co_created_consensus": observed,
        "accuracy_wilson_95_ci": wilson_interval(correct, n),
        "unweighted_cohen_kappa": kappa,
        "chance_expected_agreement": expected,
        "consensus_counts": reference_totals,
        "initial_reader_counts": reader_totals,
        "confusion_consensus_rows_initial_reader_columns": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source_hash = sha256(args.input)
    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"row_index", "cohort", "mp_label", "pg_label", "consensus_label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing the paired initial ratings or consensus")
        all_rows = list(reader)
    assert len(all_rows) == 824
    assert len({row["row_index"] for row in all_rows}) == 824
    rows = [row for row in all_rows if row["cohort"] == "Konskie"]
    assert len(rows) == 150
    statistics = {"M.P.": calculate(rows, "mp_label"), "P.G.": calculate(rows, "pg_label")}
    result = {
        "analysis": "Exploratory descriptive initial reader-versus-consensus agreement",
        "cohort": "Konskie external, n=150",
        "source_file": str(args.input.resolve()),
        "source_sha256": source_hash,
        "source_status": "Corrected initial independent pre-consensus readings confirmed by the author; consensus labels unchanged.",
        "label_order": LABELS,
        "class_order": CLASSES,
        "method": {
            "accuracy": "Number of initial labels equal to final consensus / 150.",
            "accuracy_interval": "Two-sided 95% Wilson score interval, no continuity correction; z=NormalDist().inv_cdf(0.975).",
            "kappa": "Nominal unweighted Cohen kappa; observed and chance-expected agreement from the 4x4 table.",
            "confusion_orientation": "Rows: final consensus; columns: initial reader label.",
            "interval_scope": "Binomial interval conditional on the co-created reference; assumes independent underlying cases. Does not remove dependence of the reference on reader judgments.",
        },
        "interpretation": INTERPRETATION,
        "statistics": statistics,
        "not_performed": ["Model comparison", "P-values", "Noninferiority analysis", "Clinical diagnostic accuracy assessment"],
        "source_unchanged_after_read": sha256(args.input) == source_hash,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "reader_consensus_descriptive_results.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    csv_path = args.output_dir / "reader_consensus_descriptive_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reader", "cohort", "n", "matches_consensus", "differs_from_consensus", "accuracy_against_co_created_consensus", "wilson_95_lower", "wilson_95_upper", "unweighted_cohen_kappa"])
        for name, item in statistics.items():
            writer.writerow([name, "Konskie", item["n"], item["matches_consensus"], item["differs_from_consensus"], item["accuracy_against_co_created_consensus"], *item["accuracy_wilson_95_ci"], item["unweighted_cohen_kappa"]])
    lines = [
        "Eksploracyjna zgodność ocen początkowych z konsensusem — Końskie, n = 150",
        "",
        "To opis zgodności z konsensusem współtworzonym przez tego samego radiologa. Referencja jest zależna od jego ocen (incorporation bias); nie jest to niezależny benchmark ludzki ani ocena trafności rozpoznania klinicznego.",
        "",
    ]
    for name, item in statistics.items():
        lo, hi = item["accuracy_wilson_95_ci"]
        lines.append(f"{name}: zgodność {item['matches_consensus']}/{item['n']} ({item['accuracy_against_co_created_consensus']:.1%}); 95% CI Wilsona {lo:.1%}–{hi:.1%}; nieważona κ Cohena {item['unweighted_cohen_kappa']:.3f}; rozbieżności {item['differs_from_consensus']}.")
    lines += [
        "",
        "Macierze: wiersze = konsensus końcowy; kolumny = początkowa ocena danego radiologa. Kolejność: normal, osteophytes, parasyndesmophytes, syndesmophytes. Pełne macierze i wyniki bez zaokrągleń zapisano w JSON.",
        "",
        "Przedziały Wilsona są warunkowe względem tej referencji i zakładają niezależne przypadki; nie rozwiązują problemu współtworzenia konsensusu przez ocenianego radiologa.",
        "Nie wykonywano porównań z modelami, testów istotności ani analizy równoważności/non-inferiority. Wyniki nie uzasadniają twierdzenia, że model jest lepszy, podobny lub nie gorszy od radiologa.",
        "",
        "Plik źródłowy: " + str(args.input.resolve()),
        "SHA-256 źródła: " + source_hash,
        "Plik źródłowy oraz kanoniczne pliki artykułu i repozytorium nie zostały zmienione przez ten skrypt.",
    ]
    (args.output_dir / "podsumowanie_PL.txt").write_text("\n".join(lines) + "\n")
    files = [Path(__file__).resolve(), output_json, csv_path, args.output_dir / "podsumowanie_PL.txt"]
    (args.output_dir / "sha256_manifest.json").write_text(json.dumps({"source": {"path": str(args.input.resolve()), "sha256": source_hash}, "outputs": {p.name: sha256(p) for p in files}}, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
