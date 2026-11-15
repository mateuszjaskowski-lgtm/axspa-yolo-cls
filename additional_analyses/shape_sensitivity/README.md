# Frozen-checkpoint content-ablation analysis

This supplementary analysis was completed on **1 November 2026** and is included in **materials version 2.1.0**, prepared for publication with the current manuscript's reproducibility package. It is separate from the original study training/evaluation plan. The plan was frozen after the primary and geometry/brightness results were known, before execution of this additional analysis; this was not study preregistration. Current release and deposit status are recorded in `../../release_status.json`.

## Protocol

The frozen YOLO11l-cls seed-0 checkpoint was evaluated on the same 150 external PNGs under three conditions: original images, uniform RGB gray 128, and a uniform gray equal to each image's global RGB mean rounded by `floor(mean + 0.5)`. Each entire prepared PNG was replaced before the unchanged aspect-preserving resize and centered black padding to 640 × 640. Original dimensions, the resized footprint and padding were preserved. Original absolute pixel dimensions are normalized by resizing; aspect ratio remains available. No crop, anatomical mask, retraining, checkpoint selection or threshold fitting was performed.

Before either ablation, the code required all 150 original decisions to match, exactly 138 correct classifications and a maximum absolute probability difference no greater than the frozen tolerance of 1e-4. The GPU run reproduced every original probability exactly (maximum difference 0). The earlier CPU replay exceeded the tolerance and stopped before ablations; its failed gate is retained under `prior_cpu_attempt/`. The GPU retry did not loosen the tolerance or change the frozen scientific script or plan.

## Results and limits

| Condition | Correct / 150 | Accuracy, % (Wilson 95% CI) | Balanced accuracy, % | Macro-F1 | Unweighted κ | Mean maximum probability, % |
|---|---:|---:|---:|---:|---:|---:|
| Original | 138 | 92.0 (86.5–95.4) | 93.7 | 0.927 | 0.886 | 96.21 |
| Fixed gray 128 | 55 | 36.7 (29.4–44.6) | 25.0 | 0.134 | 0.000 | 94.52 |
| Per-image mean gray | 55 | 36.7 (29.4–44.6) | 25.0 | 0.134 | 0.000 | 93.46 |

Both ablations classified **all 150 images as normal**. Their 36.7% accuracy is majority-class collapse, not random guessing. The original-minus-ablation accuracy difference was 55.33 percentage points (paired image-level bootstrap 95% CI 46.67–64.00; 10,000 replicates, seed 0) for either condition. The Wilson intervals in this ancillary analysis do not replace the primary manuscript's exact Clopper–Pearson interval of 86.4–95.8% for 138/150.

The intervention removes all spatial image content, including original background. These artificial inputs are out of distribution. Under these control conditions, the retained padding footprint alone was insufficient to reproduce the original class assignments. These controls do not establish reliance specifically on bone features. Loss of discrimination does not quantify the contribution of geometry on intact images or exclude shortcut learning. The high maximum probabilities on blank inputs also limit confidence-based interpretation. Neither 36.7% nor the prior geometry classifier's 44.7% is a universal acceptance cutoff. These results are a post hoc sensitivity analysis on already evaluated patients, not a new independent external validation.

## Contents and artifact identity

- `frozen_analysis/`: unchanged executed script, readable and JSON plans, original freeze, and its pre-existing test source. No tests or inference were rerun while preparing this supplement.
- `reference_source/`: the six exact relative source paths required by the freeze, including the manifest and original prediction NPZ. Use this directory as `--project`; do not substitute the current repository's active code.
- `gpu_results/`: three condition prediction NPZs, the 450-row CSV, summary, transformation/input audits, original replay gate, runtime, receipt and output hashes.
- `independent_integrity_audit.json`: the previously completed independent audit, with status PASS.
- `technical_provenance/`: executed Colab notebook, installer correction record and the independent audit source. The audit script was executed against the original private input package; that private package is not included here. Its saved receipt and hashes identify it.
- `source_artifact_manifest.json`: SHA-256 and byte sizes of every original analysis artifact copied into this folder. The copy process preserved all listed bytes.

The freeze SHA-256 is `2516ecb0a018afce3460e0eb0c50138aa24bb829167a354a02056d8f0590a514`; the script SHA-256 is `8c118c45efb84011207b66eed0f759df8c085688e71ebde75816ae02455ffbab`; the plan SHA-256 is `dbd6611584c31bbafbaa9b93ad3ffc06a5a7e0efa00b0d50ff65759157d664f7`; the selected checkpoint SHA-256 is `2ef70f6c73c43970cbaebed2884e71b1cd374b1545d5239abe6f2243e75c5ac6`. The freeze also records hashes for all six files in `reference_source/`.

## Reproduction with the privately held inputs

No source PNGs, checkpoint binary or private input archive are included. Obtain the original selected checkpoint and original-byte external PNGs from the corresponding author. The data root must contain exactly the registered `test/<class>/*.png` files. A later PNG representation with identical RGB pixels but different file bytes does not satisfy this frozen analysis's strict identity gate.

The observed successful environment was Python 3.13.16, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.163, Pillow 11.3.0, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.8.0, CUDA 12.8, cuDNN 91900 and NVIDIA A100-SXM4-40GB. See `gpu_results/runtime.json` and `gpu_results/gpu_runtime_target_and_observed.json`. Version/build suffixes matter; the notebook's installer correction is retained. Identical results across other environments are not guaranteed.

The following invocation uses the actual script's `run` interface and the backend settings recorded in the executed notebook. From this `shape_sensitivity` directory, substitute your private input paths and a **new, nonexistent output directory**. The scratch parent used by `--workdir` must already exist. This runs model inference; it is not needed merely to inspect saved results.

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys, runpy, torch
if not torch.cuda.is_available() or torch.version.cuda != "12.8":
    raise RuntimeError("Expected CUDA 12.8 GPU runtime")
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("highest")
sys.argv = [
    "frozen_analysis/run_shape_sensitivity.py", "run",
    "--project", "reference_source",
    "--plan", "frozen_analysis/analysis_plan.json",
    "--freeze", "frozen_analysis/analysis_freeze.json",
    "--data-root", "/path/to/original_prepared_dataset",
    "--checkpoint", "/path/to/original_selected.pt",
    "--output", "/path/to/new_reproduction_output",
    "--device", "cuda:0",
    "--workdir", "/path/to/existing_scratch_parent",
]
runpy.run_path(sys.argv[0], run_name="__main__")
```

The `run` command verifies the historical script, plan, readable plan and all six source hashes, then checks the checkpoint, all original image bytes, RGB hashes and dimensions. Do not regenerate the freeze or widen the replay tolerance after a failed run. New execution times and absolute paths are expected to differ; retain a new result directory and identify the attempt as reproduction of this already completed analysis.
