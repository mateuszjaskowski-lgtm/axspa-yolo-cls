#!/usr/bin/env python3
"""Frozen-run evaluation for four nominal spinal-radiograph classes.

No model fitting or threshold fitting on the external test set occurs here.
All requested training runs must be complete before this program starts.
Independent image-level bootstrap assumes one image per independent patient;
patient independence must be checked against source records by the investigators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import roc_auc_score, roc_curve

CLASSES = ["normal", "osteofity", "parasyndesmofity", "syndesmofity"]
MODELS = ["yolo11l-cls", "yolo26l-cls", "resnet50", "efficientnet_b0"]
IMAGE_EXTENSIONS = {".png"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean_json(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_json(row))


def ratio(num, den):
    return float(num / den) if den else float("nan")


def exact_binomial(success, total, confidence=0.95):
    """Two-sided Clopper-Pearson CI, including boundary cases."""
    if total == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "numerator": int(success), "denominator": 0, "method": "Clopper-Pearson"}
    alpha = 1 - confidence
    low = 0.0 if success == 0 else beta.ppf(alpha / 2, success, total - success + 1)
    high = 1.0 if success == total else beta.ppf(1 - alpha / 2, success + 1, total - success)
    return {"estimate": float(success / total), "ci_low": float(low), "ci_high": float(high), "numerator": int(success), "denominator": int(total), "method": "Clopper-Pearson"}


def confusion(y, predicted, n_classes=4):
    return np.bincount(np.asarray(y, int) * n_classes + np.asarray(predicted, int), minlength=n_classes ** 2).reshape(n_classes, n_classes)


def basic_scores(y, probabilities):
    y, probabilities = np.asarray(y, int), np.asarray(probabilities, float)
    predicted = probabilities.argmax(axis=1)
    cm = confusion(y, predicted)
    tp = np.diag(cm).astype(float)
    support, assigned = cm.sum(axis=1), cm.sum(axis=0)
    recalls = np.divide(tp, support, out=np.full(4, np.nan), where=support != 0)
    f1 = np.divide(2 * tp, support + assigned, out=np.full(4, np.nan), where=(support + assigned) != 0)
    accuracy = np.trace(cm) / len(y)
    chance = (support @ assigned) / len(y) ** 2
    kappa = ratio(accuracy - chance, 1 - chance)
    onehot = np.eye(4)[y]
    binary_y, binary_p = (y >= 2).astype(int), probabilities[:, 2:].sum(axis=1)
    aucs = []
    for c in range(4):
        target = (y == c).astype(int)
        aucs.append(roc_auc_score(target, probabilities[:, c]) if len(np.unique(target)) == 2 else np.nan)
    # A macro estimate is undefined when any of the four prespecified classes is absent.
    scores = {
        "accuracy": accuracy, "unweighted_cohen_kappa": kappa,
        "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1)),
        "macro_ovr_auc": float(np.mean(aucs)),
        "binary_auc": roc_auc_score(binary_y, binary_p) if len(np.unique(binary_y)) == 2 else np.nan,
        "multiclass_brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "mean_ovr_brier": float(np.mean((probabilities - onehot) ** 2)),
        "binary_brier": float(np.mean((binary_p - binary_y) ** 2)),
    }
    scores.update({f"{CLASSES[c]}_f1": f1[c] for c in range(4)})
    scores.update({f"{CLASSES[c]}_ovr_auc": aucs[c] for c in range(4)})
    return scores


def bootstrap_intervals(y, probabilities, n_bootstrap=2000, seed=42117):
    rng = np.random.default_rng(seed)
    point = basic_scores(y, probabilities)
    samples = {name: [] for name in point}
    for _ in range(n_bootstrap):
        ix = rng.integers(0, len(y), size=len(y))
        scores = basic_scores(np.asarray(y)[ix], np.asarray(probabilities)[ix])
        for name, val in scores.items():
            if np.isfinite(val):
                samples[name].append(float(val))
    result = {}
    for name, estimate in point.items():
        vals = samples[name]
        lo, hi = np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)
        degenerate = bool(len(vals) and np.ptp(vals) <= 1e-14)
        result[name] = {"estimate": estimate, "ci_low": lo, "ci_high": hi, "method": "image-level percentile bootstrap", "valid_replicates": len(vals), "requested_replicates": n_bootstrap, "bootstrap_seed": seed, "degenerate_empirical_distribution": degenerate, "uncertainty_note": "An empirical bootstrap cannot reveal unobserved error patterns. A degenerate interval, especially at perfect performance, is not evidence of certainty; inspect exact binomial class-wise intervals." if degenerate else ("No valid bootstrap replicates; metric interval undefined." if not vals else None)}
    return result


def choose_val_threshold(y, probabilities):
    target = (np.asarray(y) >= 2).astype(int)
    score = np.asarray(probabilities)[:, 2:].sum(axis=1)
    if len(np.unique(target)) != 2:
        raise ValueError("Validation threshold cannot be estimated: both binary groups are required.")
    fpr, tpr, thresholds = roc_curve(target, score, drop_intermediate=False)
    # Include the always-negative decision as a legitimate operating point, in finite form.
    thresholds = np.where(np.isinf(thresholds), np.nextafter(score.max(), np.inf), thresholds)
    youden = tpr - fpr
    max_youden = float(youden.max())
    candidates = np.flatnonzero(np.isclose(youden, max_youden, atol=1e-12, rtol=0))
    chosen = candidates[np.argmax(thresholds[candidates])]
    return {"threshold": float(thresholds[chosen]), "derivation_split": "val", "criterion": "maximize sensitivity + specificity - 1", "tie_rule": "highest threshold among maxima within absolute tolerance 1e-12", "decision_rule": "sum(P(parasyndesmofity), P(syndesmofity)) >= threshold", "max_youden": max_youden, "tied_candidate_count": int(len(candidates)), "threshold_selection_uncertainty_in_test_ci": False}


def calibration(y, probabilities, n_bins=10):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correctness = (probabilities.argmax(axis=1) == y).astype(int)
    bp, by = probabilities[:, 2:].sum(axis=1), (y >= 2).astype(int)
    result = {}
    streams = [("top_label", confidence, correctness), ("binary_morphology", bp, by)]
    streams.extend((f"ovr_{CLASSES[c]}", probabilities[:, c], (y == c).astype(int)) for c in range(4))
    for name, p, target in streams:
        rows, ece = [], 0.0
        assignment = np.minimum((p * n_bins).astype(int), n_bins - 1)
        for b in range(n_bins):
            mask = assignment == b
            n = int(mask.sum())
            observed = exact_binomial(int(target[mask].sum()), n)
            mean_p = float(np.mean(p[mask])) if n else None
            rows.append({"bin": b, "lower": b / n_bins, "upper": (b + 1) / n_bins, "n": n, "mean_probability": mean_p, "observed_fraction": observed["estimate"], "ci_low": observed["ci_low"], "ci_high": observed["ci_high"]})
            if n:
                ece += n / len(y) * abs(mean_p - observed["estimate"])
        result[name] = {"bins": rows, "ece": float(ece), "definition": f"{n_bins} equal-width bins on [0,1]; left-closed, right-open except last includes 1; empty bins not plotted; ECE weighted absolute calibration error"}
    return result


def evaluate(y, probabilities, threshold_info, n_bootstrap=2000, bootstrap_seed=42117):
    y, probabilities = np.asarray(y, int), np.asarray(probabilities, float)
    if len(y) == 0 or probabilities.shape != (len(y), 4):
        raise ValueError("Expected nonempty labels and Nx4 probabilities.")
    if not np.isfinite(probabilities).all() or probabilities.min() < -1e-8 or probabilities.max() > 1 + 1e-8:
        raise ValueError("Invalid probability values.")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Class probabilities do not sum to one; no silent renormalization is performed.")
    predicted = probabilities.argmax(axis=1)
    cm = confusion(y, predicted)
    if n_bootstrap:
        intervals = bootstrap_intervals(y, probabilities, n_bootstrap, bootstrap_seed)
    else:
        intervals = {name: {"estimate": value, "ci_low": None, "ci_high": None, "method": "point estimate; bootstrap CIs only for prespecified reference seed 0"} for name, value in basic_scores(y, probabilities).items()}
    intervals["accuracy"] = exact_binomial(int((predicted == y).sum()), len(y))
    per_class = []
    for c, name in enumerate(CLASSES):
        tp = int(cm[c, c]); fn = int(cm[c].sum() - tp); fp = int(cm[:, c].sum() - tp); tn = int(len(y) - tp - fn - fp)
        per_class.append({"class": name, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                          "sensitivity": exact_binomial(tp, tp + fn), "specificity": exact_binomial(tn, tn + fp),
                          "ppv": exact_binomial(tp, tp + fp), "npv": exact_binomial(tn, tn + fn),
                          "f1": intervals[f"{name}_f1"], "ovr_auc": intervals[f"{name}_ovr_auc"]})
    binary_y = y >= 2
    binary_prediction = probabilities[:, 2:].sum(axis=1) >= threshold_info["threshold"]
    tp = int(np.sum(binary_prediction & binary_y)); fn = int(np.sum(~binary_prediction & binary_y))
    fp = int(np.sum(binary_prediction & ~binary_y)); tn = int(np.sum(~binary_prediction & ~binary_y))
    binary = {"positive_group": CLASSES[2:], "negative_group": CLASSES[:2], "interpretation": "Radiographic morphology grouping, not a clinical axSpA diagnosis.", "threshold": threshold_info,
              "tp": tp, "fn": fn, "fp": fp, "tn": tn, "accuracy": exact_binomial(tp + tn, len(y)),
              "sensitivity": exact_binomial(tp, tp + fn), "specificity": exact_binomial(tn, tn + fp), "ppv": exact_binomial(tp, tp + fp), "npv": exact_binomial(tn, tn + fn)}
    return {"n": len(y), "class_names": CLASSES, "confusion_matrix_true_rows_predicted_columns": cm, "metrics": intervals, "per_class": per_class, "binary_morphology": binary, "calibration": calibration(y, probabilities), "ci_scope": "95% pointwise CIs conditional on fitted checkpoint and evaluation cohort sampling; test threshold treated as fixed after validation derivation. Seed variability is reported separately. These CIs do not account for model selection or threshold-estimation uncertainty.", "brier_definition": "multiclass_brier: mean(sum_c (p_c-y_c)^2), range 0..2, not divided by number of classes. mean_ovr_brier: multiclass_brier/4, the mean of four one-versus-rest Brier scores, range 0..0.5 for valid four-class probabilities. binary_brier: mean((p_positive-y_positive)^2), range 0..1. Do not interchange the conventional multiclass score and mean OvR score when updating historical manuscript values."}


def fast_comparison_scores(y, prediction):
    cm = confusion(y, prediction)
    total = len(y); true_n = cm.sum(1); pred_n = cm.sum(0); tp = cm.diagonal()
    acc = tp.sum() / total; chance = true_n @ pred_n / total ** 2
    den = true_n + pred_n
    f1 = np.divide(2 * tp, den, out=np.full(4, np.nan), where=den != 0)
    return {"accuracy": acc, "unweighted_cohen_kappa": ratio(acc - chance, 1 - chance), "macro_f1": float(np.mean(f1))}


def paired_comparison(y, p_reference, p_other, n_bootstrap=2000, seed=77011):
    y, a, b = np.asarray(y), np.asarray(p_reference).argmax(1), np.asarray(p_other).argmax(1)
    a_correct, b_correct = a == y, b == y
    a_only = int(np.sum(a_correct & ~b_correct)); b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    p = float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
    point_a, point_b = fast_comparison_scores(y, a), fast_comparison_scores(y, b)
    samples = {k: [] for k in point_a}; rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        ix = rng.integers(len(y), size=len(y))
        sa, sb = fast_comparison_scores(y[ix], a[ix]), fast_comparison_scores(y[ix], b[ix])
        for k in samples:
            value = sa[k] - sb[k]
            if np.isfinite(value): samples[k].append(value)
    differences = {}
    for key, vals in samples.items():
        lo, hi = np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)
        degenerate = bool(len(vals) and np.ptp(vals) <= 1e-14)
        differences[key] = {"estimate_reference_minus_comparator": point_a[key] - point_b[key], "ci_low": lo, "ci_high": hi, "valid_replicates": len(vals), "requested_replicates": n_bootstrap, "method": "paired image-level percentile bootstrap", "seed": seed, "degenerate_empirical_distribution": degenerate, "uncertainty_note": "Degenerate empirical interval does not establish population equivalence or certainty." if degenerate else None}
    return {"reference_correct_comparator_wrong": a_only, "reference_wrong_comparator_correct": b_only, "discordant_pairs": discordant, "mcnemar_exact_two_sided_p": p, "differences": differences}


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[ix])
        adjusted[ix] = min(1.0, running)
    return adjusted.tolist()


def generate_figures(y, probabilities, result, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    labels = ["Normal", "Osteophytes", "Parasyndesmophytes", "Syndesmophytes"]
    def save(fig, name):
        # Actual pixel resolution is rendered at both densities; PDF remains vector.
        fig.savefig(folder / f"{name}_300dpi.png", dpi=300, bbox_inches="tight")
        fig.savefig(folder / f"{name}_600dpi.png", dpi=600, bbox_inches="tight")
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    cm = confusion(y, probabilities.argmax(1))
    for normalized in [False, True]:
        shown = cm / cm.sum(1, keepdims=True) if normalized else cm
        fig, ax = plt.subplots(figsize=(6.7, 5.5), constrained_layout=True)
        im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        for r in range(4):
            for c in range(4):
                txt = f"{shown[r,c]:.1%}" if normalized else str(cm[r,c])
                ax.text(c, r, txt, ha="center", va="center", color="white" if shown[r,c] > shown.max() * .55 else "black", fontsize=12)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels, xlabel="Predicted class", ylabel="Reference class", title="Row-normalized confusion matrix" if normalized else "Confusion matrix")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.colorbar(im, ax=ax, shrink=.8)
        save(fig, "confusion_normalized" if normalized else "confusion_counts")
    fig, ax = plt.subplots(figsize=(6, 5.3), constrained_layout=True)
    for c in range(4):
        target = np.asarray(y) == c
        if np.unique(target).size == 2:
            fpr, tpr, _ = roc_curve(target, probabilities[:, c]); auc = roc_auc_score(target, probabilities[:, c])
            ax.plot(fpr, tpr, label=f"{labels[c]} (AUC {auc:.3f})")
    ax.plot([0,1], [0,1], "--", color="gray", lw=1)
    ax.set(xlim=(0,1), ylim=(0,1.01), xlabel="False-positive rate", ylabel="True-positive rate", title="One-versus-rest ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "roc_ovr")
    by = (np.asarray(y) >= 2).astype(int); bp = probabilities[:, 2:].sum(axis=1)
    if np.unique(by).size == 2:
        fig, ax = plt.subplots(figsize=(6, 5.3), constrained_layout=True)
        fpr, tpr, _ = roc_curve(by, bp)
        ax.plot(fpr, tpr, label=f"Grouped morphology (AUC {roc_auc_score(by,bp):.3f})")
        binary = result["binary_morphology"]
        ax.scatter([1 - binary["specificity"]["estimate"]], [binary["sensitivity"]["estimate"]], marker="o", label="Validation-derived operating threshold")
        ax.plot([0,1], [0,1], "--", color="gray", lw=1)
        ax.set(xlim=(0,1), ylim=(0,1.01), xlabel="False-positive rate", ylabel="True-positive rate", title="Grouped morphology ROC curve")
        ax.legend(loc="lower right", fontsize=8)
        save(fig, "roc_binary_morphology")
    fig, axs = plt.subplots(1,2,figsize=(10,4.6),constrained_layout=True)
    for ax, kind, title in zip(axs, ["top_label", "binary_morphology"], ["Top-label calibration", "Grouped morphology calibration"]):
        bins = [row for row in result["calibration"][kind]["bins"] if row["n"]]
        x = np.array([b["mean_probability"] for b in bins]); obs = np.array([b["observed_fraction"] for b in bins])
        lower = obs - np.array([b["ci_low"] for b in bins]); upper = np.array([b["ci_high"] for b in bins]) - obs
        ax.errorbar(x, obs, yerr=[lower,upper], marker="o", capsize=3)
        for b in bins:
            right = b["mean_probability"] > .8
            ax.annotate(f"n={b['n']}", (b["mean_probability"], b["observed_fraction"]), xytext=(-5 if right else 3,5), ha="right" if right else "left", textcoords="offset points", fontsize=7)
        ax.plot([0,1], [0,1], "--", color="gray", lw=1)
        ax.set(xlim=(0,1), ylim=(-.03,1.03), xlabel="Mean predicted probability", ylabel="Observed fraction", title=title)
    save(fig, "calibration")


def validate_training_freeze(training_root):
    training_root = Path(training_root)
    plan_path = training_root / "training_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("classes", plan.get("class_names")) != CLASSES:
        raise ValueError("Training plan class order differs from prespecified canonical order.")
    if set(plan["models"]) != set(MODELS) or list(plan["seeds"]) != list(range(5)):
        raise ValueError("This evaluation requires all four prespecified models and all five seeds (0..4).")
    if plan.get("reference_seed") != 0 or plan.get("external_test_previously_examined") is not True:
        raise ValueError("Training plan must identify fixed reference seed 0 and previously examined external test cohort.")
    records = []
    for model in plan["models"]:
        for seed in plan["seeds"]:
            run = training_root / model / f"seed_{seed}"
            metadata_path = run / "run_complete.json"
            if not metadata_path.exists():
                raise RuntimeError(f"External evaluation blocked: training not complete: {run}")
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("status") != "complete" or metadata.get("model") != model or int(metadata.get("seed", -1)) != seed:
                raise RuntimeError(f"Invalid training completion metadata: {run}")
            if not plan.get("protocol_sha256") or metadata.get("protocol_sha256") != plan["protocol_sha256"]:
                raise RuntimeError(f"Completed run belongs to a different frozen training protocol: {run}")
            checkpoint = run / metadata["checkpoint"]
            actual = sha256_file(checkpoint)
            if actual != metadata["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint checksum mismatch: {checkpoint}")
            if metadata.get("class_names") != CLASSES:
                raise RuntimeError(f"Checkpoint class order mismatch: {run}")
            records.append({"model": model, "seed": seed, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": actual, "run_complete_sha256": sha256_file(metadata_path), "backend": metadata["backend"], "preprocess": metadata.get("preprocess")})
    return plan, {"created_utc": datetime.now(timezone.utc).isoformat(), "training_plan_sha256": sha256_file(plan_path), "n_complete_runs": len(records), "records": records, "external_test_previously_examined": True, "reference_seed": 0, "interpretation": "Revised exploratory benchmark on an external cohort whose prior results have already been examined."}


def validate_data_manifest(data_root, manifest_path, plan):
    data_root, manifest_path = Path(data_root), Path(manifest_path)
    checksum = sha256_file(manifest_path)
    planned_checksum = plan.get("clean_manifest_sha256") or plan.get("dataset_manifest_sha256") or plan.get("development_manifest_sha256")
    if planned_checksum != checksum:
        raise RuntimeError("Dataset manifest does not match frozen training plan checksum.")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = {split: sum(row["split"] == split for row in rows) for split in ["train", "val", "test"]}
    if counts != {"train":579,"val":95,"test":150}:
        raise RuntimeError(f"Unexpected cleaned dataset counts: {counts}")
    seen_ids, seen_paths, seen_hashes = set(), set(), set()
    for row in rows:
        path = Path(row["relative_path"])
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
            raise RuntimeError(f"Invalid path in manifest: {path}")
        if path.parts[0] != row["split"] or path.parts[1] != row["class_name"] or CLASSES[int(row["label"])] != row["class_name"]:
            raise RuntimeError(f"Inconsistent folder/class/label in manifest: {path}")
        image_id = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()
        if row["image_id"] != image_id or image_id in seen_ids or path.as_posix() in seen_paths:
            raise RuntimeError(f"Duplicate/inconsistent image ID: {path}")
        actual = sha256_file(data_root / path)
        if actual != row["sha256"] or actual in seen_hashes:
            raise RuntimeError(f"Changed or duplicated image bytes: {path}")
        seen_ids.add(image_id); seen_paths.add(path.as_posix()); seen_hashes.add(actual)
    actual_files = {path.relative_to(data_root).as_posix() for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if actual_files != seen_paths:
        raise RuntimeError("Dataset image paths differ from frozen manifest.")
    return {"sha256": checksum, "counts":counts, "by_path": {row["relative_path"]:row for row in rows}}


def canonicalize_prediction_rows(rows, data_root, split, manifest=None):
    data_root = Path(data_root)
    normalized = []
    for row in rows:
        relative = Path(row["relative_path"])
        if relative.is_absolute():
            relative = relative.relative_to(data_root)
        if relative.parts[0] != split:
            relative = Path(split) / relative
        if len(relative.parts) != 3 or relative.parts[0] != split or relative.parts[1] not in CLASSES:
            raise ValueError(f"Unexpected image-relative path: {relative}")
        if ".." in relative.parts or relative.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsafe or unexpected image path: {relative}")
        label = int(row["true_label"])
        if label != CLASSES.index(relative.parts[1]):
            raise ValueError(f"Ground-truth label/folder mismatch: {relative}")
        byte_hash = sha256_file(data_root / relative)
        if row.get("sha256") and row["sha256"] != byte_hash:
            raise ValueError(f"Image hash mismatch: {relative}")
        probs = np.asarray(row["probabilities"], dtype=float)
        if probs.shape != (4,): raise ValueError("Exactly four probabilities required.")
        image_id = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
        if manifest is not None:
            frozen_row = manifest["by_path"][relative.as_posix()]
            if frozen_row["image_id"] != image_id or frozen_row["sha256"] != byte_hash:
                raise ValueError(f"Prediction-image manifest mismatch: {relative}")
        normalized.append({"image_id": image_id, "relative_path": relative.as_posix(), "sha256": byte_hash, "true_label": label, "true_class": CLASSES[label], "predicted_label": int(probs.argmax()), "predicted_class": CLASSES[int(probs.argmax())], **{f"p_{name}": float(probs[i]) for i, name in enumerate(CLASSES)}})
    normalized.sort(key=lambda row: row["relative_path"])
    expected = {p.relative_to(data_root).as_posix() for c in CLASSES for p in (data_root / split / c).glob("*.png")}
    observed = [r["relative_path"] for r in normalized]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError(f"Prediction IDs incomplete or duplicated in {split}.")
    return normalized


def seed_summary(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["model"], row["split"], row["metric"])].append(row)
    result = []
    for (model, split, metric), rows in sorted(groups.items()):
        vals = np.array([r["estimate"] for r in rows], float)
        finite = vals[np.isfinite(vals)]
        result.append({"model": model, "split": split, "metric": metric, "n_seeds": len(vals), "valid_seeds": len(finite), "mean": float(finite.mean()) if len(finite) else None, "sample_sd": float(finite.std(ddof=1)) if len(finite)>1 else None, "minimum": float(finite.min()) if len(finite) else None, "maximum": float(finite.max()) if len(finite) else None, "interpretation": "Training-seed variability on the same fixed evaluation images; seeds are not independent patients and do not increase evaluation sample size."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Defaults to data-root parent / clean_manifest.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates for the study evaluation.")
    plan, freeze = validate_training_freeze(args.training_root)
    manifest = validate_data_manifest(args.data_root, args.manifest or args.data_root.parent / "clean_manifest.csv", plan)
    freeze["clean_manifest_sha256"] = manifest["sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output_dir / "evaluation_freeze.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        if previous["training_plan_sha256"] != freeze["training_plan_sha256"] or previous["records"] != freeze["records"]:
            raise RuntimeError("Existing evaluation uses different frozen training runs. Use a new output directory and document the new analysis.")
    else:
        write_json(freeze_path, freeze)
    # Persistent lock: after this point the same training directory cannot be used
    # to fit or select further models after seeing external evaluation results.
    lock_path = args.training_root / "external_evaluation_started.json"
    lock = {"training_plan_sha256": freeze["training_plan_sha256"], "clean_manifest_sha256": manifest["sha256"], "records": freeze["records"], "external_test_previously_examined": True}
    if lock_path.exists():
        previous_lock = json.loads(lock_path.read_text())
        if previous_lock != lock:
            raise RuntimeError("Training weights, protocol or manifest changed after external evaluation was locked.")
    else:
        write_json(lock_path, lock)
    from train_models import predict_split
    outputs, long_metrics, overview = {}, [], []
    for model in plan["models"]:
        for seed in plan["seeds"]:
            run = args.training_root / model / f"seed_{seed}"
            run_output = args.output_dir / model / f"seed_{seed}"
            arrays = {}
            threshold = None
            # The freeze has already verified every training run before either split is evaluated.
            for split in ["val", "test"]:
                print(f"Evaluating {model} seed={seed} split={split}", flush=True)
                rows = predict_split(run, args.data_root / split, batch_size=args.batch_size, device=args.device)
                rows = canonicalize_prediction_rows(rows, args.data_root, split, manifest)
                ids = np.asarray([r["image_id"] for r in rows], dtype=str)
                hashes = np.asarray([r["sha256"] for r in rows], dtype=str)
                y = np.asarray([r["true_label"] for r in rows], dtype=int)
                p = np.asarray([[r[f"p_{c}"] for c in CLASSES] for r in rows], dtype=np.float64)
                run_output.mkdir(parents=True, exist_ok=True)
                write_csv(run_output / f"{split}_predictions.csv", rows)
                np.savez_compressed(run_output / f"{split}_predictions.npz", image_id=ids, sha256=hashes, y_true=y, probabilities=p, class_names=np.asarray(CLASSES, dtype=str))
                arrays[split] = (ids, hashes, y, p)
                if split == "val":
                    threshold = choose_val_threshold(y, p)
                    write_json(run_output / "validation_selected_threshold.json", threshold)
            for split, (ids, hashes, y, p) in arrays.items():
                results = evaluate(y, p, threshold, args.bootstrap_replicates if seed == 0 else 0, bootstrap_seed=42117 + (0 if split == "val" else 1))
                results.update({"model": model, "training_seed": seed, "split": split, "checkpoint_sha256": sha256_file(run / json.loads((run / "run_complete.json").read_text())["checkpoint"])})
                write_json(run_output / f"{split}_metrics.json", results)
                for kind, values in results["calibration"].items(): write_csv(run_output / f"{split}_calibration_{kind}.csv", values["bins"])
                for metric, value in results["metrics"].items():
                    long_metrics.append({"model": model, "seed": seed, "split": split, "n": len(y), "metric": metric, "estimate": value["estimate"], "ci_low": value["ci_low"], "ci_high": value["ci_high"], "ci_method": value["method"]})
                overview.append({"model": model, "seed": seed, "split": split, "n": len(y), **{k:v["estimate"] for k,v in results["metrics"].items() if k in ["accuracy","unweighted_cohen_kappa","balanced_accuracy","macro_f1","macro_ovr_auc","binary_auc","multiclass_brier","mean_ovr_brier","binary_brier"]}})
                if seed == 0:
                    generate_figures(y, p, results, run_output / f"{split}_figures")
                    outputs[(model, split)] = (ids, hashes, y, p)
    comparisons = []
    reference = outputs[("yolo11l-cls", "test")]
    for comparator in [m for m in plan["models"] if m != "yolo11l-cls"]:
        other = outputs[(comparator, "test")]
        if not all(np.array_equal(reference[i], other[i]) for i in range(3)):
            raise ValueError("Pairwise comparison blocked: IDs, hashes, or labels are not identically aligned.")
        result = paired_comparison(reference[2], reference[3], other[3], args.bootstrap_replicates)
        result.update({"reference": "yolo11l-cls", "comparator": comparator, "seed": 0, "split": "test", "n": len(reference[2]), "exploratory": True})
        comparisons.append(result)
    adjusted = holm_adjust([r["mcnemar_exact_two_sided_p"] for r in comparisons])
    for result, p in zip(comparisons, adjusted): result["mcnemar_holm_adjusted_p"] = p
    write_json(args.output_dir / "paired_comparisons_seed0.json", {"family": "Three prespecified YOLO11-versus-comparator tests on the external cohort; exploratory", "difference_intervals": "Pointwise 95% intervals; not simultaneous or multiplicity-adjusted.", "results": comparisons})
    write_csv(args.output_dir / "metrics_all_runs.csv", long_metrics)
    write_csv(args.output_dir / "all_runs_overview.csv", overview)
    write_csv(args.output_dir / "five_seed_summary.csv", seed_summary(long_metrics))
    write_json(args.output_dir / "analysis_complete.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "training_plan_sha256": freeze["training_plan_sha256"], "clean_manifest_sha256": manifest["sha256"], "evaluation_script_sha256": sha256_file(__file__), "n_evaluated_runs": len(plan["models"]) * len(plan["seeds"]), "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_scope": "All four models, validation and external test, reference seed 0 only. Other seeds receive point estimates and exact binomial CIs for proportions, plus five-seed summary statistics.", "python": platform.python_version(), "numpy": np.__version__, "cohort_independence": "Image-level uncertainty assumes one independent patient per image; investigators must verify source-patient identity and cross-split separation.", "primary_reference_model": "yolo11l-cls", "primary_reference_seed": 0, "external_test_previously_examined": True})
    print(f"Completed evaluation: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
