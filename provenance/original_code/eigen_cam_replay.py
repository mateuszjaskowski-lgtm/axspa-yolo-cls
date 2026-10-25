#!/usr/bin/env python3
"""Auditable, optional Eigen-CAM for the newly trained primary YOLO11 model.

This is a NEW qualitative analysis. It does not reconstruct historical Figure 5.
The same official YOLO.predict forward pass supplies probabilities, the exact
preprocessed input tensor, and spatial features. Overlays are registered to that
input after the frozen whole-PNG, aspect-preserving resize and square padding.

No model training, model selection, test-set tuning, or manual source-image
preprocessing is performed by this module. This additive repair replays the
frozen evaluator's complete ordered batches; it does not replace saved evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CLASS_NAMES = ["normal", "osteofity", "parasyndesmofity", "syndesmofity"]
DISPLAY_NAMES = ["Normal", "Osteophytes", "Parasyndesmophytes", "Syndesmophytes"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    """Hash shape, dtype, and C-order bytes so tensor hashes are unambiguous."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(array.shape), "dtype": array.dtype.str},
                             sort_keys=True).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def eigen_cam_from_activations(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Class-agnostic first principal-component projection of C×H×W features.

    Center each feature channel over spatial positions. SVD sign is fixed by
    requiring the largest-absolute loading in the first right singular vector
    to be positive (first channel wins an exact tie). Negative projections are
    clipped to zero and the nonnegative projection is scaled to [0, 1]. This
    sign convention is deterministic and does not use predictions or labels.
    Degenerate/constant features yield an all-zero map, explicitly recorded.
    """
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim != 3 or min(a.shape) < 1 or a.shape[1] * a.shape[2] < 2:
        raise ValueError(f"Expected spatial C×H×W activations, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("Non-finite activations; no CAM was generated")
    c, h, w = a.shape
    matrix = a.reshape(c, h * w).T.copy()
    matrix -= matrix.mean(axis=0, keepdims=True)
    if not np.any(matrix):
        zero = np.zeros((h, w), dtype=np.float32)
        return zero, zero.copy(), {
            "degenerate": True, "reason": "all channels spatially constant",
            "leading_singular_values": [], "first_pc_variance_fraction": 0.0,
            "sign_anchor_channel": None, "raw_projection_min": 0.0,
            "raw_projection_max": 0.0,
        }
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    component = vh[0].copy()
    anchor = int(np.argmax(np.abs(component)))
    if component[anchor] < 0:
        component *= -1
    projection = (matrix @ component).reshape(h, w)
    positive = np.maximum(projection, 0.0)
    maximum = float(positive.max())
    cam = positive / maximum if maximum > 0 else np.zeros_like(positive)
    denominator = float(np.sum(singular * singular))
    meta = {
        "degenerate": maximum == 0.0,
        "leading_singular_values": singular[:5].tolist(),
        "first_pc_variance_fraction": float(singular[0] ** 2 / denominator) if denominator else 0.0,
        "sign_anchor_channel": anchor,
        "raw_projection_min": float(projection.min()),
        "raw_projection_max": float(projection.max()),
        "first_two_singular_values_nearly_tied": bool(
            len(singular) > 1 and np.isclose(singular[0], singular[1], rtol=1e-7, atol=1e-12)
        ),
    }
    return cam.astype(np.float32), projection.astype(np.float32), meta


def choose_examples(rows: list[dict], max_errors: int = 4) -> list[dict]:
    """Correct: first stable path per reference class. Errors: class then fill.

    No confidence-based choice; labels only define correctly/incorrectly
    classified strata. At most one correct example per class. Errors first
    cover each available reference class, then fill by stable relative path,
    with no duplicate images. No missing examples are fabricated.
    """
    if not 0 <= max_errors <= 4:
        raise ValueError("max_errors must be between 0 and 4")
    ordered = sorted(rows, key=lambda r: r["relative_path"])
    if len({r["relative_path"] for r in ordered}) != len(ordered):
        raise ValueError("Duplicate relative paths in evaluation rows")
    selected: list[dict] = []
    for label in range(len(CLASS_NAMES)):
        candidates = [r for r in ordered if r["true_label"] == label and r["pred_label"] == label]
        if candidates:
            selected.append({**candidates[0], "selection_group": "correct"})
    errors = [r for r in ordered if r["true_label"] != r["pred_label"]]
    chosen_errors = []
    for label in range(len(CLASS_NAMES)):
        candidates = [r for r in errors if r["true_label"] == label]
        if candidates and len(chosen_errors) < max_errors:
            chosen_errors.append(candidates[0])
    used = {r["relative_path"] for r in chosen_errors}
    for row in errors:
        if len(chosen_errors) >= max_errors:
            break
        if row["relative_path"] not in used:
            chosen_errors.append(row)
            used.add(row["relative_path"])
    selected.extend({**r, "selection_group": "incorrect"} for r in chosen_errors)
    return selected


def safe_image_path(dataset_root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe relative image path: {relative_path}")
    resolved = (dataset_root / rel).resolve()
    if not resolved.is_relative_to(dataset_root.resolve()):
        raise ValueError(f"Image escapes dataset root: {relative_path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def normalization_details(transforms: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Only invert an inspectable, deterministic torchvision inference chain."""
    from spine_transforms import SpineResizePad, TRANSFORM_PROTOCOL
    from torchvision import transforms as T

    chain = getattr(transforms, "transforms", None)
    if chain is None:
        raise ValueError("Cannot inspect predictor transforms; refusing an unregistered overlay")
    names = [type(t).__name__ for t in chain]
    expected_types = [SpineResizePad, T.ToTensor, T.Normalize]
    if [type(t) for t in chain] != expected_types:
        raise ValueError(f"Expected the frozen SpineResizePad, ToTensor, Normalize inference chain, got {names}")
    if type(chain[0].size) is not int or chain[0].size != TRANSFORM_PROTOCOL["size"] or chain[0].size != 640:
        raise ValueError("Eigen-CAM requires the frozen 640-pixel whole-PNG resize-and-pad transform")
    mean = np.asarray(chain[2].mean, dtype=np.float32)
    std = np.asarray(chain[2].std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("Expected three-channel normalization with positive standard deviations")
    expected_normalization = TRANSFORM_PROTOCOL["yolo_normalization"]
    if (not np.array_equal(mean, np.asarray(expected_normalization["mean"], dtype=np.float32))
            or not np.array_equal(std, np.asarray(expected_normalization["std"], dtype=np.float32))):
        raise ValueError("Loaded YOLO normalization differs from the frozen transform protocol")
    return mean, std, names


def tensor_to_rgb(input_chw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    a = np.asarray(input_chw, dtype=np.float32)
    if a.ndim != 3 or a.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW model tensor, got {a.shape}")
    rgb = (a * std[:, None, None] + mean[:, None, None]).transpose(1, 2, 0)
    if not np.all(np.isfinite(rgb)) or rgb.min() < -1e-4 or rgb.max() > 1.0001:
        raise ValueError("Inverse-normalized input is outside [0,1]; refusing to change image contrast")
    return np.clip(rgb, 0, 1)


def inference_environment(device: str = "cuda:0") -> dict:
    """Observe numerical settings without changing those used by evaluation."""
    import torch
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch_cuda": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
        "note": "Recorded during this replay; historical evaluation did not record these numerical flags. No flags changed by the CAM repair.",
    }
    for package in ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "matplotlib"):
        result[package] = importlib.metadata.version(package)
    if torch.cuda.is_available() and str(device) != "cpu":
        d = torch.device("cuda:" + str(device) if str(device).isdigit() else device)
        result.update(gpu=torch.cuda.get_device_name(d),
                      gpu_total_bytes=torch.cuda.get_device_properties(d).total_memory,
                      gpu_capability=list(torch.cuda.get_device_capability(d)))
    return result


def capture_prediction_batch(yolo: Any, source_paths: list[Path], target: Any,
                             backend_model: Any, predict_kwargs: dict,
                             capture_indices: list[int]) -> tuple:
    """Capture chosen slices from one full evaluation-shaped forward pass.

    Predictor warmup must finish before calling this helper. Every returned
    probability is checked against its input path; hooks never run an extra
    model forward and copy only selected activation/input slices to CPU.
    """
    import torch
    count = len(source_paths)
    indices = sorted(set(capture_indices))
    if count < 1 or any(i < 0 or i >= count for i in indices):
        raise ValueError("Invalid batch/capture indices")
    captured_input, captured_features, input_shapes, feature_shapes = {}, {}, [], []

    def input_hook(module: Any, args: tuple) -> None:
        tensor = args[0]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[0] != count:
            raise TypeError("Expected the complete BCHW evaluation batch at the model input")
        if tensor.dtype != torch.float32:
            raise TypeError("Locked evaluation and CAM require float32 model inputs")
        input_shapes.append(list(tensor.shape))
        for i in indices:
            captured_input[i] = tensor[i].detach().float().cpu().numpy().copy()

    def feature_hook(module: Any, args: tuple, output: Any) -> None:
        if (not isinstance(output, torch.Tensor) or output.ndim != 4
                or output.shape[0] != count or min(output.shape[2:]) < 2):
            raise TypeError("Expected complete BCHW spatial features matching the input batch")
        feature_shapes.append(list(output.shape))
        for i in indices:
            captured_features[i] = output[i].detach().float().cpu().numpy().copy()

    before = backend_model.register_forward_pre_hook(input_hook)
    after = target.register_forward_hook(feature_hook)
    try:
        backend_model.eval()
        with torch.inference_mode():
            results = yolo.predict(source=[str(p) for p in source_paths], **predict_kwargs)
    finally:
        before.remove()
        after.remove()
    if len(input_shapes) != 1 or len(feature_shapes) != 1 or len(results) != count:
        raise RuntimeError("Expected exactly one full evaluation batch/forward pass; refusing ambiguous CAM provenance")
    probabilities = []
    for path, result in zip(source_paths, results):
        if Path(result.path).resolve() != path.resolve():
            raise RuntimeError("YOLO result path/order mismatch during CAM replay")
        if result.probs is None:
            raise RuntimeError("Model did not return classification probabilities")
        vector = result.probs.data.detach().float().cpu().numpy()
        if vector.shape != (4,):
            raise RuntimeError("Expected four nominal-class probabilities")
        probabilities.append(vector)
    captured = {i: {"input": captured_input[i], "features": captured_features[i]} for i in indices}
    return np.asarray(probabilities), captured, input_shapes[0], feature_shapes[0]


def verify_batched_replay(yolo: Any, dataset_root: Path, rows: list[dict],
                          selected: list[dict], target: Any, backend_model: Any,
                          kwargs: dict, output_root: Path, provenance: dict | None) -> tuple:
    """Replay all frozen-order batches and check all images before rendering.

    This does not overwrite evaluation predictions. On disagreement it writes
    a diagnostic report, then stops without declaring the CAM package complete.
    """
    batch_size = int(kwargs["batch"])
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    selected_paths = {r["relative_path"] for r in selected}
    if len(selected_paths) != len(selected) or not selected_paths.issubset({r["relative_path"] for r in rows}):
        raise ValueError("Selected paths must be unique members of the locked cohort")
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "in_progress", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)), "evaluation_provenance": provenance or {},
        "prediction_settings": dict(kwargs), "environment": inference_environment(str(kwargs.get("device", "cuda:0"))),
        "probability_atol": 1e-4, "probability_rtol": 1e-4, "argmax_must_match": True,
        "cohort_size": len(rows), "batch_size": batch_size,
        "order": "Frozen evaluator image_records class/name lexicographic order",
        "evaluation_artifacts_replaced": False, "batches": [], "images": [],
    }
    captures = {}
    try:
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            paths = [safe_image_path(dataset_root, r["relative_path"]) for r in part]
            indices = [i for i, row in enumerate(part) if row["relative_path"] in selected_paths]
            probabilities, captured, input_shape, feature_shape = capture_prediction_batch(
                yolo, paths, target, backend_model, kwargs, indices)
            batch_id = len(report["batches"])
            members = [{"relative_path": r["relative_path"], "image_id": r["image_id"],
                        "sha256": r["sha256"]} for r in part]
            report["batches"].append({"batch_index_zero_based": batch_id,
                "cohort_start_index_zero_based": start, "actual_batch_size": len(part),
                "model_input_shape_bchw": input_shape, "feature_shape_bchw": feature_shape,
                "members": members, "captured_local_indices": indices})
            for i, (row, vector) in enumerate(zip(part, probabilities)):
                expected = np.asarray(row["probabilities"], dtype=np.float64)
                finite = bool(np.isfinite(vector).all())
                delta = float(np.max(np.abs(vector.astype(np.float64) - expected))) if finite else None
                predicted = int(vector.argmax()) if finite else None
                probability_match = bool(finite and np.allclose(vector, expected, atol=1e-4, rtol=1e-4))
                argmax_match = predicted == int(row["pred_label"])
                check = {"relative_path": row["relative_path"], "image_id": row["image_id"],
                    "sha256": row["sha256"], "cohort_index_zero_based": start+i,
                    "batch_index_zero_based": batch_id, "index_within_batch_zero_based": i,
                    "evaluation_probabilities": expected.tolist(),
                    "replay_probabilities": [float(v) if np.isfinite(v) else None for v in vector],
                    "maximum_probability_difference": delta, "probabilities_match": probability_match,
                    "evaluation_predicted_label": int(row["pred_label"]),
                    "replay_predicted_label": predicted, "argmax_matches": argmax_match,
                    "passed": bool(probability_match and argmax_match),
                    "selected_for_cam": row["relative_path"] in selected_paths}
                report["images"].append(check)
                if i in captured:
                    captures[row["relative_path"]] = {**captured[i], "probabilities": vector.copy(),
                        "batch_index_zero_based": batch_id, "index_within_batch_zero_based": i,
                        "actual_batch_size": len(part), "batch_members": members,
                        "model_input_shape_bchw": input_shape, "feature_shape_bchw": feature_shape,
                        "maximum_probability_difference": delta}
        report.update(status="passed" if all(r["passed"] for r in report["images"]) else "failed",
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      n_verified_images=len(report["images"]),
                      n_failed_images=sum(not r["passed"] for r in report["images"]))
        report["maximum_probability_difference"] = max(
            (r["maximum_probability_difference"] for r in report["images"]
             if r["maximum_probability_difference"] is not None), default=None)
        if set(captures) != selected_paths:
            raise RuntimeError("Not all selected CAM images were captured during cohort replay")
    except Exception as exc:
        report.update(status="error", completed_utc=datetime.now(timezone.utc).isoformat(),
                      error_type=type(exc).__name__, error=str(exc), n_verified_images=len(report["images"]))
        write_json(output_root / "replay_verification.json", report)
        raise
    write_json(output_root / "replay_verification.json", report)
    if report["status"] != "passed":
        raise RuntimeError(f"CAM full-batch replay disagrees with locked evaluation for {report['n_failed_images']} images; "
                           f"see {output_root / 'replay_verification.json'}. No relabeling or tolerance relaxation performed.")
    return captures, report


def render_figure(examples: list[dict], output_root: Path, dpi: int) -> list[str]:
    """Render real image/map pixels at the requested resolution; no fake DPI relabeling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [[e for e in examples if e["selection_group"] == key]
              for key in ("correct", "incorrect")]
    groups = [group for group in groups if group]
    columns = max(len(group) for group in groups)
    figure = plt.figure(figsize=(columns * 4.0, len(groups) * 2.70 + 0.85), layout="constrained")
    outer = figure.add_gridspec(len(groups), columns)
    letters = iter("abcdefgh")
    for row_index, group in enumerate(groups):
        for column_index, example in enumerate(group):
            inner = outer[row_index, column_index].subgridspec(1, 2, wspace=0.03)
            left = figure.add_subplot(inner[0, 0])
            right = figure.add_subplot(inner[0, 1])
            left.imshow(example["input_rgb"], interpolation="nearest")
            right.imshow(example["overlay"], interpolation="nearest")
            for axis in (left, right):
                axis.set_axis_off()
            letter = next(letters)
            example["panel"] = letter
            status = "Correct" if example["selection_group"] == "correct" else "Incorrect"
            left.set_title(f"({letter}) {status}: reference\n{DISPLAY_NAMES[example['true_label']]}"
                           "\nActual model input", fontsize=8, loc="left")
            right.set_title(f"Eigen-CAM: predicted\n{DISPLAY_NAMES[example['pred_label']]}"
                            f"\np = {example['probabilities'][example['pred_label']]:.3f}", fontsize=8)
    figure.suptitle("Exploratory Eigen-CAM\nNew primary YOLO11 model" if columns == 1 else
                    "Exploratory Eigen-CAM: new primary YOLO11 model", fontsize=13)
    footer = ("Actual whole-PNG resize + square-padding input.\n"
              "Class-agnostic PCA maps; illustrative examples,\nnot causal explanations."
              if columns == 1 else
              "Both images show the same whole-PNG, aspect-preserving resize + square-padding input.\n"
              "Class-agnostic PCA maps; illustrative examples, not causal explanations.")
    figure.supxlabel(footer, fontsize=8)
    png, pdf = output_root / f"eigen_cam_figure_{dpi}dpi.png", output_root / "eigen_cam_figure.pdf"
    figure.savefig(png, dpi=dpi, facecolor="white")
    figure.savefig(pdf, dpi=dpi, facecolor="white")
    plt.close(figure)
    return [png.name, pdf.name]


def validate_locked_inputs(dataset_root: Path, weights: Path, expected_weight_sha256: str,
                           rows: list[dict], batch_size: int) -> None:
    """Validate provenance and image bytes without loading a model or touching CUDA."""
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if sha256_file(weights) != expected_weight_sha256:
        raise ValueError("Selected weights do not match the recorded training/evaluation hash")
    if not rows:
        raise ValueError("The complete external evaluation cohort is empty")
    for row in rows:
        required = {"relative_path", "image_id", "sha256", "true_label", "pred_label", "probabilities"}
        if required - row.keys():
            raise ValueError(f"Prediction row lacks {sorted(required - row.keys())}")
        if row["true_label"] not in range(4) or row["pred_label"] not in range(4):
            raise ValueError("Invalid four-class evaluation label")
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        if probabilities.shape != (4,) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Invalid evaluation probability vector")
        if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1, atol=1e-4):
            raise ValueError("Evaluation probabilities must be nonnegative and sum to one")
        if int(np.argmax(probabilities)) != row["pred_label"]:
            raise ValueError("Saved predicted class is inconsistent with its probabilities")
        source = safe_image_path(dataset_root, row["relative_path"])
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"Source image hash mismatch: {row['relative_path']}")
    from train_models import image_records
    enumerated = image_records(dataset_root / "test")
    expected_order = ["test/" + r["relative_path"] for r in enumerated]
    if [r["relative_path"] for r in rows] != expected_order:
        raise ValueError("Locked evaluation row order differs from frozen evaluator image_records order")
    for row, observed in zip(rows, enumerated):
        if row["sha256"] != observed["sha256"] or row["true_label"] != observed["true_label"]:
            raise ValueError("Frozen evaluation/source image metadata disagree")
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")


def generate_eigen_cam(*, dataset_root: Path, weights: Path, expected_weight_sha256: str,
                       rows: list[dict], output_root: Path, imgsz: int,
                       device: str = "0", dpi: int = 600,
                       evaluation_provenance: dict | None = None,
                       training_run: Path | None = None, batch_size: int = 32) -> dict:
    """Create a self-contained qualitative audit from already evaluated cases.

    Row contract: relative_path, image_id, sha256, true_label, pred_label,
    probabilities (a four-element list in CLASS_NAMES order). The entire
    evaluation cohort must be passed so selection cannot be confidence-filtered.
    """
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    import torch
    from ultralytics import YOLO

    if dpi not in (300, 600):
        raise ValueError("Use an actual export resolution of 300 or 600 dpi")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty; use a new directory: {output_root}")
    validate_locked_inputs(dataset_root, weights, expected_weight_sha256, rows, batch_size)
    selected = choose_examples(rows)
    if not selected:
        raise ValueError("No examples available")

    if training_run is not None:
        from train_models import load_for_inference, ULTRALYTICS_VERSION
        if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
            raise RuntimeError(f"Use the locked Ultralytics {ULTRALYTICS_VERSION} environment")
        yolo, _, backend = load_for_inference(training_run, device=device)
        if backend != "ultralytics":
            raise ValueError("Eigen-CAM is configured only for the primary YOLO11 checkpoint")
    else:
        raise ValueError("A frozen completed training run is required; standalone fused YOLO loading is prohibited")
    names = yolo.names
    actual_names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if actual_names != CLASS_NAMES:
        raise ValueError(f"Checkpoint class order mismatch: {actual_names}")
    # Preserve the frozen evaluator's complete batches, including its short final batch.
    kwargs = dict(imgsz=int(imgsz), device=device, half=False, augment=False,
                  batch=int(batch_size), stream=False, save=False, verbose=False)
    # Prime the same unfused backend with the FIRST COMPLETE evaluation chunk.
    # Pinned Ultralytics warms up on this real input batch before its first prediction.
    # Register hooks afterwards so that only the actual replay pass is captured.
    prime_files = [str(safe_image_path(dataset_root, r["relative_path"])) for r in rows[:batch_size]]
    yolo.predict(source=prime_files, **kwargs)
    backend_model = getattr(yolo.predictor.model, "model", None)
    if backend_model is None:
        raise RuntimeError("Cannot inspect the initialized PyTorch prediction backend")
    blocks = getattr(backend_model, "model", None)
    if blocks is None or len(blocks) < 2 or type(blocks[-1]).__name__ != "Classify":
        raise RuntimeError("Expected primary YOLO11 classification model ending in Classify")
    target = blocks[-2]
    target_names = [name for name, module in backend_model.named_modules() if module is target]
    if len(target_names) != 1:
        raise RuntimeError("Could not uniquely identify the target spatial block")
    target_name = target_names[0]
    target_type = f"{type(target).__module__}.{type(target).__qualname__}"
    mean, std, transform_names = normalization_details(yolo.predictor.transforms)
    print(f"Eigen-CAM target layer: {target_name} ({target_type})")
    print("Overlay region: actual whole-PNG resize + square-padding tensor seen by the model")
    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    captures, replay_report = verify_batched_replay(
        yolo, dataset_root, rows, selected, target, backend_model, kwargs,
        output_root, evaluation_provenance)
    # Nothing is rendered until every member of the frozen cohort has passed.
    panels_dir = output_root / "panels"
    panels_dir.mkdir()
    rendered, records = [], []
    for order, row in enumerate(selected, start=1):
        source = safe_image_path(dataset_root, row["relative_path"])
        captured = captures[row["relative_path"]]
        probabilities, x, activations = captured["probabilities"], captured["input"], captured["features"]
        expected = np.asarray(row["probabilities"], dtype=np.float64)
        max_difference = captured["maximum_probability_difference"]
        rgb = tensor_to_rgb(x, mean, std)
        cam_native, raw_projection, pca_meta = eigen_cam_from_activations(activations)
        resized = np.asarray(Image.fromarray(cam_native).resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
        # Zero/constant CAMs remain uncolored; no artificial activation is shown.
        heat_rgb = colormaps["turbo"](resized)[..., :3].astype(np.float32)
        overlay = rgb.copy() if pca_meta["degenerate"] else (0.60 * rgb + 0.40 * heat_rgb)
        stem = f"panel_{order:02d}_{row['image_id'][:12]}"
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(panels_dir / f"{stem}_input.png")
        Image.fromarray(np.rint(overlay * 255).astype(np.uint8)).save(panels_dir / f"{stem}_overlay.png")
        np.savez_compressed(panels_dir / f"{stem}_arrays.npz",
                            model_input=x, spatial_activations=activations,
                            raw_pc_projection=raw_projection,
                            cam_native=cam_native, cam_resized=resized,
                            probabilities=probabilities)
        with Image.open(source) as original:
            source_size = list(original.size)
        record = {
            "selection_order": order, "selection_group": row["selection_group"],
            "relative_path": row["relative_path"], "image_id": row["image_id"],
            "source_sha256": row["sha256"], "source_size_width_height": source_size,
            "true_label": row["true_label"], "true_class": CLASS_NAMES[row["true_label"]],
            "pred_label": int(probabilities.argmax()), "pred_class": CLASS_NAMES[int(probabilities.argmax())],
            "probabilities": probabilities.tolist(), "evaluation_probabilities": expected.tolist(),
            "maximum_probability_difference_from_evaluation": max_difference,
            "replayed_batch": {key: captured[key] for key in (
                "batch_index_zero_based", "index_within_batch_zero_based", "actual_batch_size",
                "batch_members", "model_input_shape_bchw", "feature_shape_bchw")},
            "weight_sha256": expected_weight_sha256,
            "target_layer": target_name, "target_layer_type": target_type,
            "model_input_shape_chw": list(x.shape), "model_input_sha256": array_digest(x),
            "activation_shape_chw": list(activations.shape), "activation_sha256": array_digest(activations),
            "pca": pca_meta,
            "input_png": f"panels/{stem}_input.png", "overlay_png": f"panels/{stem}_overlay.png",
            "raw_arrays_npz": f"panels/{stem}_arrays.npz",
        }
        records.append(record)
        rendered.append({**row, "input_rgb": rgb, "overlay": overlay, "probabilities": probabilities})
    files = render_figure(rendered, output_root, dpi)
    for record, example in zip(records, rendered):
        record["panel"] = example["panel"]
    environment = inference_environment(device)
    report = {
        "status": "complete", "analysis": "New exploratory Eigen-CAM analysis of newly trained primary YOLO11",
        "created_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "weights_filename": weights.name, "weights_sha256": expected_weight_sha256,
        "class_names": CLASS_NAMES, "environment": environment,
        "device": str(next(backend_model.parameters()).device), "dtype": "float32",
        "prediction_settings": kwargs,
        "full_cohort_replay": {"report": "replay_verification.json",
            "report_sha256": sha256_file(output_root / "replay_verification.json"),
            "n_verified_images": replay_report["n_verified_images"],
            "maximum_probability_difference": replay_report["maximum_probability_difference"],
            "batch_size": int(batch_size), "status": replay_report["status"]},
        "target_layer_policy": "Final top-level spatial block immediately before the Classify head; frozen before visualization",
        "target_layer": target_name, "target_layer_type": target_type,
        "preprocessing": {
            "source": "Official loaded YOLO ClassificationPredictor, same forward pass as saved CAM probabilities",
            "transforms_repr": repr(yolo.predictor.transforms), "transform_names": transform_names,
            "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
            "overlay_coordinates": "Actual model-input tensor after whole-PNG aspect-preserving resize and square padding; inverse normalization only for display",
            "additional_manual_source_cropping_or_enhancement_in_this_module": False,
            "augmentation_at_inference": False,
        },
        "eigen_cam_method": {
            "class_agnostic": True, "gradients_used": False,
            "pca": "float64 SVD of spatial-position × feature-channel matrix, centered per feature channel",
            "projection": "Centered activations projected onto the first right singular vector",
            "sign_convention": "Loading with greatest absolute magnitude is made positive; first channel breaks exact tie",
            "normalization": "Clip negative projection to zero, divide by spatial maximum; constant maps remain zero",
            "resampling": "Bilinear interpolation to exact input tensor dimensions",
            "smoothing": "No Gaussian, augmentation, or eigen-smoothing; only bilinear display resampling",
            "overlay": "turbo color map on [0,1], 40% heatmap + 60% inverse-normalized model input",
            "limitations": "Illustrative feature patterns; not class-specific causal explanations, localization validation, or proof against shortcut learning",
        },
        "selection": {
            "cohort_size": len(rows), "correct_count": sum(r["true_label"] == r["pred_label"] for r in rows),
            "incorrect_count": sum(r["true_label"] != r["pred_label"] for r in rows),
            "rule": "Correct: first lexicographic relative_path per reference class. Errors: first per available reference class, then remaining lexicographic paths, maximum four, no duplicate path. No confidence-based selection.",
            "selected_count": len(records), "absent_strata": "No image fabricated when a correct class or error stratum is absent",
        },
        "evaluation_provenance": evaluation_provenance or {}, "panels": records, "figure_files": files,
    }
    write_json(output_root / "eigen_cam_provenance.json", report)
    (output_root / "figure_caption.txt").write_text(
        "Exploratory Eigen-CAM visualizations from the newly trained primary YOLO11l-cls "
        "checkpoint (prespecified reference seed 0). Correctly classified examples are the "
        "first image by lexicographic relative path within each available reference class. "
        "Incorrectly classified examples first represent each available reference class, "
        "then remaining paths up to four examples; no confidence-based selection was used. "
        "The paired panels display the actual image region supplied to the model after its "
        "automatic whole-PNG, aspect-preserving resize and square-padding transform, "
        "without and with the feature-map overlay. "
        "Displayed probabilities and feature activations were obtained in the same forward "
        "pass, using the same complete ordered image batches as the locked evaluation. "
        "All external-cohort probability vectors were checked before rendering. "
        "The maps are the first principal-component projections of the last top-level "
        "spatial feature block before the classification head. They are class-agnostic "
        "qualitative illustrations, not causal explanations, validated lesion localizations, "
        "or evidence excluding shortcut learning. Source identifiers, hashes, selection, "
        "layer, normalization and software details are recorded in the accompanying audit.\n",
        encoding="utf-8")
    columns = ["panel", "selection_group", "relative_path", "image_id", "source_sha256",
               "true_class", "pred_class", *[f"p_{name}" for name in CLASS_NAMES],
               "weight_sha256", "target_layer", "model_input_sha256", "activation_sha256",
               "input_png", "overlay_png", "raw_arrays_npz"]
    with (output_root / "eigen_cam_panels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            flat = {key: record[key] for key in columns if key in record}
            flat.update({f"p_{name}": record["probabilities"][i] for i, name in enumerate(CLASS_NAMES)})
            writer.writerow(flat)
    outputs = {str(path.relative_to(output_root)): sha256_file(path)
               for path in sorted(output_root.rglob("*")) if path.is_file()}
    write_json(output_root / "artifact_hashes.json", outputs)
    print(f"Saved {len(records)} newly generated Eigen-CAM examples to {output_root}")
    return report


def load_locked_artifacts(dataset_root: Path, training_root: Path,
                          evaluation_root: Path) -> tuple[Path, dict, list[dict], dict]:
    """Accept only completed evaluation of the fixed primary model and seed 0."""
    complete_path = evaluation_root / "analysis_complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (complete.get("status") != "complete"
            or complete.get("primary_reference_model") != "yolo11l-cls"
            or complete.get("primary_reference_seed") != 0
            or complete.get("n_evaluated_runs") != 20):
        raise ValueError("Completed evaluation of primary YOLO11l-cls seed 0 is required")
    run = training_root / "yolo11l-cls" / "seed_0"
    metadata_path = run / "run_complete.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("status") != "complete" or metadata.get("model") != "yolo11l-cls"
            or metadata.get("seed") != 0 or metadata.get("class_names") != CLASS_NAMES
            or metadata.get("backend") != "ultralytics"):
        raise ValueError("Primary model training metadata do not match the prespecified reference run")
    freeze_path = evaluation_root / "evaluation_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_runs = {(m, s) for m in ("yolo11l-cls", "yolo26l-cls", "resnet50", "efficientnet_b0")
                     for s in range(5)}
    frozen_runs = [(r["model"], r["seed"]) for r in freeze.get("records", [])]
    if len(frozen_runs) != 20 or set(frozen_runs) != expected_runs or freeze.get("n_complete_runs") != 20:
        raise ValueError("All 20 prespecified training runs must be present in the completed evaluation freeze")
    if freeze.get("training_plan_sha256") != complete.get("training_plan_sha256"):
        raise ValueError("Evaluation freeze/analysis-completion plan hashes differ")
    if sha256_file(training_root / "training_plan.json") != complete["training_plan_sha256"]:
        raise ValueError("Training plan changed since evaluation")
    frozen = [r for r in freeze["records"] if r["model"] == "yolo11l-cls" and r["seed"] == 0]
    if (len(frozen) != 1 or frozen[0]["checkpoint_sha256"] != metadata["checkpoint_sha256"]
            or frozen[0]["run_complete_sha256"] != sha256_file(metadata_path)):
        raise ValueError("Primary checkpoint or completion metadata changed since evaluation")
    checkpoint_relative = Path(metadata["checkpoint"])
    if checkpoint_relative.is_absolute() or ".." in checkpoint_relative.parts:
        raise ValueError("Unsafe checkpoint-relative path")
    if int(metadata["preprocess"]["imgsz"]) != 640:
        raise ValueError("Primary study model must use the locked 640-pixel preprocessing")
    prediction_path = evaluation_root / "yolo11l-cls" / "seed_0" / "test_predictions.csv"
    metrics_path = prediction_path.parent / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
        raise ValueError("External evaluation metrics do not belong to the selected checkpoint")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        rel = Path(raw["relative_path"])
        if len(rel.parts) != 3 or rel.parts[0] != "test" or rel.parts[1] not in CLASS_NAMES:
            raise ValueError("Expected test/<class>/<image>.png prediction paths")
        true_label, pred_label = int(raw["true_label"]), int(raw["predicted_label"])
        if true_label != CLASS_NAMES.index(rel.parts[1]):
            raise ValueError("Reference class differs from image-folder class")
        if hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest() != raw["image_id"]:
            raise ValueError("Image identifier does not match its frozen relative path")
        rows.append({"relative_path": rel.as_posix(), "image_id": raw["image_id"],
                     "sha256": raw["sha256"], "true_label": true_label, "pred_label": pred_label,
                     "probabilities": [float(raw[f"p_{name}"]) for name in CLASS_NAMES]})
    paths = [r["relative_path"] for r in rows]
    expected = {p.relative_to(dataset_root).as_posix()
                for name in CLASS_NAMES for p in (dataset_root / "test" / name).glob("*.png")}
    if len(paths) != 150 or len(set(paths)) != len(paths) or set(paths) != expected:
        raise ValueError("The complete, uniquely identified external cohort of 150 images is required")
    npz_path = prediction_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if (arrays["class_names"].tolist() != CLASS_NAMES
                or arrays["image_id"].tolist() != [r["image_id"] for r in rows]
                or arrays["sha256"].tolist() != [r["sha256"] for r in rows]
                or arrays["y_true"].tolist() != [r["true_label"] for r in rows]
                or not np.array_equal(arrays["probabilities"], np.asarray([r["probabilities"] for r in rows]))):
            raise ValueError("Evaluation CSV and NPZ disagree")
    provenance = {"primary_model": "yolo11l-cls", "reference_seed": 0,
                  "analysis_complete_sha256": sha256_file(complete_path),
                  "evaluation_freeze_sha256": sha256_file(freeze_path),
                  "test_predictions_csv_sha256": sha256_file(prediction_path),
                  "test_predictions_npz_sha256": sha256_file(npz_path),
                  "test_metrics_sha256": sha256_file(metrics_path),
                  "run_complete_sha256": sha256_file(metadata_path),
                  "training_plan_sha256": complete["training_plan_sha256"],
                  "external_test_previously_examined": complete.get("external_test_previously_examined"),
                  "note": "Seed and layer are fixed in advance; no best-external-seed selection"}
    return run, metadata, rows, provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--data-root", dest="dataset_root", type=Path, required=True,
                        help="Prepared clean dataset containing train/, val/ and test/")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, choices=[300, 600], default=600)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Exact batch size of the completed frozen evaluation; never tune this value")
    parser.add_argument("--validate-only", action="store_true",
                        help="Verify frozen weights/source images/provenance only; no GPU/model forward or outputs")
    parser.add_argument("--frozen-code-root", type=Path, required=True,
                        help="Original frozen code directory, read-only; contains evaluator/trainer/transforms")
    args = parser.parse_args(argv)
    frozen_code_root = args.frozen_code_root.resolve()
    for name in ("train_models.py", "spine_transforms.py", "evaluate_models.py", "eigen_cam.py"):
        if not (frozen_code_root / name).is_file():
            raise FileNotFoundError(frozen_code_root / name)
    sys.path.insert(0, str(frozen_code_root))
    run, metadata, rows, provenance = load_locked_artifacts(
        args.dataset_root, args.training_root, args.evaluation_root)
    provenance["frozen_code_sha256"] = {
        name: sha256_file(frozen_code_root / name) for name in (
            "train_models.py", "spine_transforms.py", "evaluate_models.py", "eigen_cam.py")}
    provenance["replay_script_sha256"] = sha256_file(Path(__file__))
    protocol_path = args.training_root.parent / "protocol_artifacts_frozen.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        frozen_batch = protocol.get("artifacts", {}).get("evaluation_config", {}).get("batch_size")
        if frozen_batch is None or int(frozen_batch) != args.batch_size:
            raise ValueError("Requested replay batch size differs from frozen evaluation_config.batch_size")
        provenance["protocol_artifacts_frozen_sha256"] = sha256_file(protocol_path)
        provenance["batch_size_source"] = "protocol_artifacts_frozen.json artifacts.evaluation_config.batch_size"
    else:
        provenance["batch_size_source"] = "Explicit CLI/default; protocol_artifacts_frozen.json unavailable"
    provenance["evaluation_batch_size"] = args.batch_size
    provenance["correction_scope"] = "CAM capture now uses exact evaluation batch composition; training and evaluation are read-only"
    if args.validate_only:
        validate_locked_inputs(args.dataset_root, run / metadata["checkpoint"],
                               metadata["checkpoint_sha256"], rows, args.batch_size)
        print(json.dumps({"status": "locked_inputs_verified", "n": len(rows),
                          "weight_sha256": metadata["checkpoint_sha256"],
                          "provenance": provenance}, sort_keys=True, allow_nan=False))
        return
    generate_eigen_cam(dataset_root=args.dataset_root, weights=run / metadata["checkpoint"],
                       expected_weight_sha256=metadata["checkpoint_sha256"], rows=rows,
                       output_root=args.output_root, imgsz=metadata["preprocess"]["imgsz"],
                       device=args.device, dpi=args.dpi,
                       evaluation_provenance=provenance, training_run=run,
                       batch_size=args.batch_size)


if __name__ == "__main__":
    main()
