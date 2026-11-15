#!/usr/bin/env python3
"""Post hoc, inference-only content ablation for the frozen Diagnostics YOLO11 seed 0.

No training, checkpoint selection, binary threshold adjustment or image cropping.
Run `freeze` before `run`; a failed identity/replay gate stops before ablation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["normal", "osteofity", "parasyndesmofity", "syndesmofity"]
CONDITIONS = ["original", "fixed_gray_128", "per_image_mean_gray"]
CHECKPOINT_SHA256 = "2ef70f6c73c43970cbaebed2884e71b1cd374b1545d5239abe6f2243e75c5ac6"
N_EXTERNAL = 150
BASELINE_CORRECT = 138
PROBABILITY_ATOL = 1e-4
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 0
REQUIRED_CORE = {"torch": "2.11.0", "torchvision": "0.26.0", "ultralytics": "8.4.163", "Pillow": "11.3.0"}
SOURCE_FILES = ["code/train_models.py", "code/spine_transforms.py", "code/study_design.py",
                "data/clean_manifest.csv", "analysis/yolo11l-cls/seed_0/test_predictions.npz",
                "runs/yolo11l-cls/seed_0/selection_metadata.json"]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def rgb_sha256(image):
    image = image.convert("RGB")
    return hashlib.sha256(f"RGB:{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def gray_image(image, condition):
    """Uniform RGB8 content BEFORE the unchanged archived resize/pad transform."""
    rgb = image.convert("RGB")
    if condition == "fixed_gray_128":
        value = 128
    elif condition == "per_image_mean_gray":
        # Float64 RGB mean, all pixels/channels; round half upward, not bankers' rounding.
        value = int(math.floor(float(np.asarray(rgb, dtype=np.float64).mean()) + 0.5))
    else:
        raise ValueError(f"Not a prespecified ablation: {condition}")
    return Image.new("RGB", rgb.size, (value, value, value)), value


def frozen_configuration():
    return {"conditions": CONDITIONS, "checkpoint_sha256": CHECKPOINT_SHA256,
            "n_external": N_EXTERNAL, "baseline_correct": BASELINE_CORRECT,
            "baseline_probability_absolute_tolerance": PROBABILITY_ATOL,
            "all_baseline_argmax_must_match": True, "batch_size": 32, "imgsz": 640,
            "half": False, "augment": False, "conv_batchnorm_fusion": False,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "gray_rounding": "floor(float64_mean_all_original_RGB_pixels_and_channels + 0.5)",
            "confidence_interval": "two-sided 95% Wilson accuracy; paired image-level percentile bootstrap accuracy difference",
            "required_core_versions_without_build_suffix": REQUIRED_CORE,
            "input_identity_mode": "strict_original_PNG_bytes_plus_independent_RGB_and_dimensions",
            "center_crop_or_stretch": False, "new_training_or_parameter_selection": False,
            "interpretation": "Post hoc OOD content ablation, not causal attribution or proof of bone-specific reliance."}


def freeze(args):
    project, plan, output = Path(args.project).resolve(), Path(args.plan).resolve(), Path(args.freeze).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze: {output}")
    json.loads(plan.read_text())  # Reject malformed or incomplete plan before freezing.
    sources = {relative: sha256(project / relative) for relative in SOURCE_FILES}
    output.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "created_utc": utcnow(), "stage": "before_any_new_study_image_predictions",
             "prior_information": "Primary experiment and geometry/brightness analysis already known; this is a post hoc protocol, not preregistration.",
             "script_sha256": sha256(__file__), "plan_sha256": sha256(plan),
             "readable_plan_sha256": sha256(plan.with_name("PLAN_ANALIZY.txt")),
             "source_hashes": sources, "configuration": frozen_configuration()}
    write_json(output, value)
    print(json.dumps({"freeze": str(output), "sha256": sha256(output), "inference_performed": False}))


def verify_freeze(project, plan, lock_path):
    lock = json.loads(Path(lock_path).read_text())
    if lock["script_sha256"] != sha256(__file__) or lock["plan_sha256"] != sha256(plan):
        raise RuntimeError("Frozen script or plan changed. Do not recreate the lock after seeing results.")
    if lock["readable_plan_sha256"] != sha256(Path(plan).with_name("PLAN_ANALIZY.txt")):
        raise RuntimeError("Readable analysis plan changed after freeze.")
    if lock["configuration"] != frozen_configuration():
        raise RuntimeError("Prespecified configuration differs from lock.")
    for relative, expected in lock["source_hashes"].items():
        if sha256(Path(project) / relative) != expected:
            raise RuntimeError(f"Frozen project artifact changed: {relative}")
    return lock


def manifest_test_rows(project):
    with (Path(project) / "data/clean_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    if len(rows) != N_EXTERNAL or len({r["image_id"] for r in rows}) != N_EXTERNAL:
        raise RuntimeError("External manifest is not 150 unique images.")
    paths = [r["relative_path"] for r in rows]
    if len(set(paths)) != len(paths):
        raise RuntimeError("Duplicate manifest path.")
    for row in rows:
        rel = Path(row["relative_path"])
        if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 3 or rel.parts[0] != "test":
            raise RuntimeError("Unsafe/unexpected external manifest path.")
        if CLASSES[int(row["label"])] != row["class_name"] or rel.parts[1] != row["class_name"]:
            raise RuntimeError("Manifest class mapping inconsistent.")
    return rows


def align_reference(manifest, archived):
    """Return frozen probabilities in manifest order; ID + byte hash + label must all agree."""
    ids = [str(x) for x in archived["image_id"]]
    if len(ids) != len(manifest) or len(set(ids)) != len(ids):
        raise RuntimeError("Archived IDs duplicate or incomplete.")
    if list(archived["class_names"]) != CLASSES:
        raise RuntimeError("Archived class ordering changed.")
    lookup = {key: i for i, key in enumerate(ids)}
    if set(lookup) != {row["image_id"] for row in manifest}:
        raise RuntimeError("Reference IDs differ from manifest.")
    order = []
    for row in manifest:
        i = lookup[row["image_id"]]
        if str(archived["sha256"][i]) != row["sha256"] or int(archived["y_true"][i]) != int(row["label"]):
            raise RuntimeError("Reference image hash or label mismatch.")
        order.append(i)
    return np.asarray(archived["probabilities"])[order]


def verify_inputs(data_root, manifest):
    root = Path(data_root).resolve()
    found = {p.relative_to(root).as_posix() for p in (root / "test").rglob("*.png")}
    expected = {row["relative_path"] for row in manifest}
    if found != expected:
        raise RuntimeError(f"External file set differs: missing {sorted(expected-found)[:5]}, extra {sorted(found-expected)[:5]}")
    audit = []
    for row in manifest:
        path = root / row["relative_path"]
        actual_bytes = sha256(path)
        with Image.open(path) as image:
            image.load()
            actual_rgb = rgb_sha256(image)
            if image.size != (int(row["width"]), int(row["height"])):
                raise RuntimeError(f"Dimensions differ: {path}")
        if actual_rgb != row["independent_rgb_sha256"]:
            raise RuntimeError(f"Decoded RGB pixels differ: {path}")
        byte_match = actual_bytes == row["sha256"]
        if not byte_match:
            raise RuntimeError(f"Original PNG bytes differ: {path}. Use the exact source PNG, even if decoded RGB matches.")
        audit.append(dict(image_id=row["image_id"], relative_path=row["relative_path"],
                          expected_file_sha256=row["sha256"], actual_file_sha256=actual_bytes,
                          file_bytes_match=byte_match, independent_rgb_sha256=actual_rgb,
                          decoded_rgb_and_dimensions_match=True))
    return audit


def align_predictions(manifest, prediction_rows):
    by_path = {r["relative_path"]: r for r in prediction_rows}
    if len(by_path) != len(prediction_rows) or len(by_path) != len(manifest):
        raise RuntimeError("Prediction paths duplicated/incomplete.")
    expected_paths = {r["relative_path"].removeprefix("test/") for r in manifest}
    if set(by_path) != expected_paths:
        raise RuntimeError("Prediction path set differs from the frozen manifest.")
    ordered = []
    for row in manifest:
        pred = by_path[row["relative_path"].removeprefix("test/")]
        if int(pred["true_label"]) != int(row["label"]) or pred["class_name"] != row["class_name"]:
            raise RuntimeError("Prediction label mapping changed.")
        ordered.append(pred["probabilities"])
    p = np.asarray(ordered, dtype=float)
    if p.shape != (len(manifest), 4) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise RuntimeError("Invalid prediction probabilities.")
    return p


def baseline_gate(y, probabilities, reference):
    if probabilities.shape != (N_EXTERNAL, 4) or reference.shape != probabilities.shape:
        raise RuntimeError("Baseline gate requires exactly 150 four-class probability vectors.")
    argmax_matches = int(np.sum(probabilities.argmax(1) == reference.argmax(1)))
    correct = int(np.sum(probabilities.argmax(1) == y))
    max_abs = float(np.max(np.abs(probabilities - reference)))
    passed = argmax_matches == N_EXTERNAL and correct == BASELINE_CORRECT and max_abs <= PROBABILITY_ATOL
    return {"status": "PASS" if passed else "FAIL", "argmax_matches": argmax_matches,
            "correct": correct, "total": N_EXTERNAL, "maximum_absolute_probability_difference": max_abs,
            "mean_absolute_probability_difference": float(np.abs(probabilities-reference).mean()),
            "prespecified_absolute_tolerance": PROBABILITY_ATOL,
            "exact_probability_array_match": bool(np.array_equal(probabilities, reference))}


def wilson(correct, total):
    z = 1.959963984540054
    p = correct / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1-p) / total + z*z/(4*total*total)) / den
    return {"estimate": p, "ci_low": center-half, "ci_high": center+half,
            "confidence": .95, "method": "Wilson", "correct": int(correct), "total": int(total)}


def metrics(y, probabilities):
    pred = probabilities.argmax(1)
    cm = np.bincount(y * 4 + pred, minlength=16).reshape(4, 4)
    true_n, pred_n = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    recalls = tp / true_n
    f1 = np.divide(2 * tp, true_n + pred_n, out=np.zeros(4, dtype=float), where=true_n + pred_n > 0)
    acc = float(np.trace(cm) / len(y))
    expected = float(true_n @ pred_n / len(y)**2)
    return {"accuracy": wilson(int(np.trace(cm)), len(y)), "balanced_accuracy": float(recalls.mean()),
            "macro_f1": float(f1.mean()), "unweighted_cohen_kappa": float((acc-expected)/(1-expected)),
            "confusion_matrix_true_rows_predicted_columns": cm.tolist(), "true_class_counts": true_n.tolist(),
            "predicted_class_counts": pred_n.tolist(), "class_sensitivity": recalls.tolist(),
            "mean_max_probability": float(probabilities.max(1).mean())}


def paired_difference(y, original, variant):
    d = (original.argmax(1) == y).astype(float) - (variant.argmax(1) == y).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(y), size=(BOOTSTRAP_REPLICATES, len(y)))
    samples = d[indices].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {"estimate_original_minus_variant": float(d.mean()), "ci_low": float(low), "ci_high": float(high),
            "confidence": .95, "method": "paired image-level percentile bootstrap",
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "degenerate": bool(np.ptp(samples) == 0),
            "prediction_agreement_with_original": float(np.mean(original.argmax(1) == variant.argmax(1))),
            "prediction_agreement_count_with_original": int(np.sum(original.argmax(1) == variant.argmax(1))),
            "caveat": "Conditional on checkpoint and this OOD intervention; not a causal effect or proof of equivalence."}


def write_prediction_outputs(output, manifest, probabilities):
    y = np.asarray([int(r["label"]) for r in manifest])
    for condition, p in probabilities.items():
        np.savez_compressed(output / f"{condition}_predictions.npz",
            image_id=np.asarray([r["image_id"] for r in manifest]),
            sha256=np.asarray([r["sha256"] for r in manifest]), y_true=y,
            probabilities=p, class_names=np.asarray(CLASSES))
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_id", "relative_path", "y_true", "condition", "prediction", "correct"] + [f"p_{c}" for c in CLASSES]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, p in probabilities.items():
            for row, vector in zip(manifest, p):
                item = dict(image_id=row["image_id"], relative_path=row["relative_path"], y_true=int(row["label"]),
                            condition=condition, prediction=int(vector.argmax()), correct=bool(vector.argmax()==int(row["label"])))
                item.update({f"p_{c}": float(vector[i]) for i,c in enumerate(CLASSES)})
                writer.writerow(item)


def environment():
    import torch
    versions = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "ultralytics", "Pillow", "numpy", "scipy", "scikit-learn"]}
    for name, expected in REQUIRED_CORE.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"Core runtime mismatch: {name}={versions[name]}, required {expected}. No silent substitution.")
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "runtime_note": "Recorded for this sensitivity run. Historical evaluation runtime flags were not independently recorded."}


def run(args):
    project, plan, output = Path(args.project).resolve(), Path(args.plan).resolve(), Path(args.output).resolve()
    lock = verify_freeze(project, plan, args.freeze)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    checkpoint = Path(args.checkpoint).resolve()
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Wrong checkpoint bytes.")
    manifest = manifest_test_rows(project)
    input_audit = verify_inputs(args.data_root, manifest)
    archived_path = project / "analysis/yolo11l-cls/seed_0/test_predictions.npz"
    with np.load(archived_path, allow_pickle=False) as archived:
        reference = align_reference(manifest, archived)
    y = np.asarray([int(r["label"]) for r in manifest], dtype=int)
    if int(np.sum(reference.argmax(1) == y)) != BASELINE_CORRECT:
        raise RuntimeError("Archived reference does not contain 138/150 correct predictions.")
    runtime = environment()
    sys.path.insert(0, str(project / "code"))
    from train_models import predict_split, load_for_inference
    from spine_transforms import SpineResizePad
    output.mkdir(parents=True)
    write_json(output / "input_audit.json", input_audit)
    write_json(output / "runtime.json", dict(runtime, requested_device=args.device, created_utc=utcnow()))
    shutil.copy2(args.freeze, output / "analysis_freeze.json")
    shutil.copy2(plan, output / "analysis_plan.json")
    predictions, transformation_audit = {}, []
    try:
        with tempfile.TemporaryDirectory(prefix="axspa_shape_", dir=args.workdir) as temp_name:
            temp = Path(temp_name)
            run_dir = temp / "run"
            run_dir.mkdir()
            # A temporary exact copy avoids changing public project or original run metadata.
            shutil.copy2(checkpoint, run_dir / "selected.pt")
            selection = json.loads((project / "runs/yolo11l-cls/seed_0/selection_metadata.json").read_text())
            if selection["checkpoint_sha256"] != CHECKPOINT_SHA256 or selection["backend"] != "ultralytics":
                raise RuntimeError("Selected-run metadata inconsistent with frozen checkpoint.")
            selection["checkpoint"] = "selected.pt"
            write_json(run_dir / "selection_metadata.json", selection)
            checked_model, checked_transform, checked_backend = load_for_inference(run_dir, device=args.device)
            operations = getattr(checked_transform, "transforms", [])
            if (checked_backend != "ultralytics" or len(operations) != 3
                    or not isinstance(operations[0], SpineResizePad) or operations[0].size != 640
                    or type(operations[1]).__name__ != "ToTensor"
                    or type(operations[2]).__name__ != "Normalize"
                    or list(operations[2].mean) != [0, 0, 0] or list(operations[2].std) != [1, 1, 1]):
                raise RuntimeError("Loaded checkpoint transform does not match the frozen whole-PNG protocol.")
            spatial_transform = operations[0]
            write_json(output / "loaded_transform_audit.json", {"transform_repr": repr(checked_transform),
                "backend": checked_backend, "whole_png_aspect_preserving_pad_640_verified": True})
            del checked_model
            print("Replaying unmodified external inputs...", flush=True)
            rows = predict_split(run_dir, Path(args.data_root) / "test", batch_size=32, device=args.device)
            predictions["original"] = align_predictions(manifest, rows)
            gate = baseline_gate(y, predictions["original"], reference)
            write_json(output / "baseline_replay_gate.json", gate)
            write_prediction_outputs(output, manifest, predictions)
            if gate["status"] != "PASS":
                raise RuntimeError("Baseline replay failed. Ablations were NOT run. Do not loosen the tolerance after this result.")
            for condition in CONDITIONS[1:]:
                variant_root = temp / condition
                for row in manifest:
                    source = Path(args.data_root) / row["relative_path"]
                    destination = variant_root / row["relative_path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(source) as image:
                        image.load()
                        source_mean = float(np.asarray(image.convert("RGB"), dtype=np.float64).mean())
                        uniform, value = gray_image(image, condition)
                        width, height, left, top = spatial_transform.geometry(image.size)
                        transformed = np.asarray(spatial_transform(uniform))
                        expected_canvas = np.zeros((640, 640, 3), dtype=np.uint8)
                        expected_canvas[top:top+height, left:left+width] = value
                        if not np.array_equal(transformed, expected_canvas):
                            raise RuntimeError("Ablated footprint/intensity/padding differs from the frozen transform.")
                        uniform.save(destination, format="PNG")
                        variant_rgb_hash = rgb_sha256(uniform)
                    transformation_audit.append(dict(image_id=row["image_id"], relative_path=row["relative_path"],
                        condition=condition, uniform_gray=value, width=int(row["width"]), height=int(row["height"]),
                        original_mean_rgb_unrounded=source_mean,
                        resized_width=width, resized_height=height, padding_left=left, padding_top=top,
                        padding_right=640-left-width, padding_bottom=640-top-height,
                        unchanged_footprint_and_zero_padding_verified=True,
                        variant_file_sha256=sha256(destination), variant_independent_rgb_sha256=variant_rgb_hash))
                print(f"Evaluating prespecified condition {condition}...", flush=True)
                rows = predict_split(run_dir, variant_root / "test", batch_size=32, device=args.device)
                predictions[condition] = align_predictions(manifest, rows)
                write_prediction_outputs(output, manifest, predictions)
                write_json(output / "transformation_audit.json", transformation_audit)
        results = {condition: metrics(y, p) for condition, p in predictions.items()}
        comparisons = {condition: paired_difference(y, predictions["original"], predictions[condition]) for condition in CONDITIONS[1:]}
        with (project / "data/clean_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
            train_counts = np.bincount([int(r["label"]) for r in csv.DictReader(stream) if r["split"] == "train"], minlength=4)
        majority_label = int(train_counts.argmax())
        if int(train_counts.sum()) != 579 or majority_label != 0:
            raise RuntimeError("Frozen training majority baseline differs from prespecified normal-class comparator.")
        summary = {"status": "COMPLETE", "created_utc": utcnow(), "study": "Diagnostics four-class spinal morphology",
            "analysis_type": "post hoc exploratory frozen-checkpoint inference-only content ablation", "n": len(y),
            "class_order": CLASSES, "conditions": results, "paired_comparisons": comparisons,
            "descriptive_comparators": {"training_majority_class_normal_accuracy": 55/150,
                "training_class_counts": train_counts.tolist(), "majority_class_derived_from_train": CLASSES[majority_label],
                "training_majority_class_normal_balanced_accuracy": .25,
                "previous_geometry_brightness_logistic_accuracy": 67/150,
                "note": "Neither majority accuracy nor previous logistic accuracy is a universal chance or acceptance cutoff."},
            "interpretation_limits": ["Ablated inputs are out of distribution.",
                "Low ablation accuracy does not prove use of bone-specific features or absence of shortcuts on original images.",
                "High ablation accuracy shows information exploitable under the intervention, not its causal contribution on intact images.",
                "Aspect ratio is retained; original absolute pixel height and width are removed by normalization apart from rounding/interpolation.",
                "No retraining, threshold fitting, checkpoint selection, cropping or post-result parameter choice performed."],
            "freeze_sha256": sha256(args.freeze), "plan_sha256": lock["plan_sha256"], "checkpoint_sha256": CHECKPOINT_SHA256}
        write_json(output / "summary.json", summary)
        hashes = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
        write_json(output / "output_hashes.json", hashes)
        print(json.dumps({k: v["accuracy"] for k, v in results.items()}, indent=2))
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAILED", "created_utc": utcnow(), "error": str(exc),
            "completed_conditions": list(predictions), "do_not_interpret_as_complete_analysis": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["freeze", "run"]:
        command = commands.add_parser(name)
        command.add_argument("--project", required=True)
        command.add_argument("--plan", required=True)
        command.add_argument("--freeze", required=True)
    commands.choices["freeze"].set_defaults(function=freeze)
    execute = commands.choices["run"]
    execute.add_argument("--data-root", required=True, help="Folder containing test/<class>/*.png")
    execute.add_argument("--checkpoint", required=True)
    execute.add_argument("--output", required=True, help="Must not already exist")
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--workdir", help="Existing scratch parent for temporary derived images")
    execute.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
