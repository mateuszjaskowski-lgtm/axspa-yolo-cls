# Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model

Companion materials for **Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model**, by Mateusz Jaśkowski, Paweł Górski, Zbigniew Guzera and Michał Podgórski, prepared for *Diagnostics*.

**Current repository materials: version 2.0.0.** See [RELEASE_NOTES.md](RELEASE_NOTES.md) and the [GitHub release](https://github.com/mateuszjaskowski-lgtm/axspa-yolo-cls/releases/tag/v2.0.0). Code and non-image reproducibility materials are archived in Zenodo at [DOI: 10.5281/zenodo.23649843](https://doi.org/10.5281/zenodo.23649843).

Reproducibility materials for a retrospective two-center study of **four nominal morphology classes** on prepared lateral spinal radiographs. The reference model is **YOLO11l-cls, seed 0**. YOLO26l-cls, ResNet50 and EfficientNet-B0 are comparison configurations, each evaluated with seeds 0–4.

This repository describes the completed cleaned-data experiment. In a clarification dated **2026-10-06**, the author confirmed that the external cohort had not been used before the final evaluation. The study is retrospective; prospective clinical utility has not been established. The original analysis plan was agreed verbally, without prospective registration; the archived computational plan for the cleaned-data experiment is a separate record.

## Data and labels

| Split | Normal | Osteophytes | Parasyndesmophytes | Syndesmophytes | Total |
|---|---:|---:|---:|---:|---:|
| Training, Łódź | 211 | 210 | 81 | 77 | 579 |
| Validation, Łódź | 37 | 34 | 12 | 12 | 95 |
| External test, Końskie | 55 | 54 | 20 | 21 | 150 |

The accepted dataset contains **824 images**. The author clarified on **2026-10-06** that the development split was made manually in Windows Explorer: within each class folder, PNGs were sorted numerically and contiguous ranges were moved to validation until approximately 15% of that class had been selected. The blocks were selected without regard to image content, before the first training, and assignments were not changed in response to model outcomes. This was a manual, nonrandom allocation; no `scikit-learn` split or random-allocation seed was used. Numerical filename order does not establish acquisition chronology. After the documented removal of duplicate training copies, validation contains **95/674 development images (14.1%)**; class-specific fractions are given in [`data/README.md`](data/README.md). The authoritative retained memberships and image hashes are in [`data/clean_manifest.csv`](data/clean_manifest.csv).

Class order is fixed: `normal`, `osteofity`, `parasyndesmofity`, `syndesmofity`. Integer labels 0–3 identify categories and do not represent severity grades. The study's syndesmophyte category represents bridging lesions under its reference-labeling protocol.

M.J. prepared the source PNGs with Windows Snipping Tool, outlining the examined spinal segment with surrounding tissues before radiologist labeling. Computational preprocessing preserves the complete prepared PNG field, resizes proportionally with bilinear interpolation, and adds centered black padding to 640 × 640 pixels. Training uses horizontal flips and brightness/contrast jitter. Source preparation and training augmentation are separate steps.

## Current results

The reference YOLO11l-cls seed-0 model classified **138/150 external images correctly: 92.0% accuracy (exact 95% CI 86.4–95.8%)**, with macro-F1 0.9265, unweighted Cohen's κ 0.8862 and macro one-versus-rest AUC 0.9875.

| Configuration | Seed-0 external accuracy | Five-seed mean ± sample SD |
|---|---:|---:|
| YOLO11l-cls, reference | 92.00% | 88.40% ± 3.52 percentage points |
| YOLO26l-cls | 87.33% | 89.47% ± 1.28 percentage points |
| ResNet50 | 93.33% | 91.07% ± 3.35 percentage points |
| EfficientNet-B0 | 90.67% | 89.60% ± 1.12 percentage points |

Seed 0 was specified for the revision benchmark before its evaluation; the best external seed is not selected retrospectively. Five seeds measure training variability on the **same 150 images** and do not increase the number of independent patients. The three seed-0 accuracy comparisons against YOLO11 were not significant after Holm adjustment; this does not establish equivalence or imply that every secondary metric has a null difference. Detailed results are in [`analysis/`](analysis/).

## Repository contents

- `code/`: active preparation, training, evaluation and transform scripts; the corrected batch-consistent Eigen-CAM replay; lightweight independent metrics checks. Descriptive study-design fields have been corrected without changing numerical methods. The reported maps were produced with `eigen_cam_replay.py`; the original `eigen_cam.py` is required only for the frozen provenance checks.
- `notebooks/axspa_clean_retraining.ipynb`: reproduction notebook with corrected explanations and embedded active script payloads. Its executable cells are not byte-identical to the originally executed notebook. Original code and hash-bound correction records are preserved separately.
- `analysis/`: 40 validation/test prediction CSVs and matching numeric NPZs, metric JSONs, validation-selected thresholds, paired comparisons, five-seed summaries and reader-agreement materials.
- `runs/`: frozen plan and run/selection metadata, without checkpoint binaries.
- `data/clean_manifest.csv`: current 824-image split, technical identifiers and hashes, with no patient names or clinical demographics.
- `checkpoints/manifest.json`: hashes, sizes and archive membership of all 20 selected checkpoints.
- `protocol_artifacts_frozen.json`: the byte-exact frozen protocol at the location expected by the replay script.
- `eigen_cam/`: the eight-panel source ledger, original CAM provenance, all-150 replay verification and artifact hashes; metadata only, without medical image pixels or feature/input arrays.
- `review_supplement/cam_audit.json`: the recorded independent CAM consistency checks used by the publication-layout script.
- `study_design.json`: current author-confirmed allocation and external-cohort-use description.
- `provenance/`: original executed assets in `original_code/`, exact-source hashes, technical-repair records, notebook code-integrity records and the hash-bound `study_design_correction.json` correction ledger.
- `environment/`: observed environment and full recorded Colab package listing.

The repository contains **no source PNGs, checkpoint binaries or CAM image pixels**. PNG images are available from the corresponding author, [Mateusz Jaśkowski](mailto:mjaskowski@zoz.konskie.pl), on reasonable request. Checkpoints are separate research artifacts; use the checkpoint manifest to identify and verify the exact files, and contact the corresponding author if you do not have the associated archives. The code license does not grant rights to restricted source images or third-party model assets.

## Recompute metrics without images or weights

The following checks all 40 prediction tables against the accompanying point estimates:

```bash
python -m pip install -r requirements-analysis.txt
python code/recompute_public_metrics.py
```

The initial reader-agreement results can also be reproduced from the ordered, nonidentifying rating pairs:

```bash
python code/recompute_reader_agreement.py
```

`data/reader_ratings.csv` contains only row order, center and the two initial labels plus consensus. The accompanying script reproduces the saved agreement estimates and all three 10,000-resample bootstrap intervals; it does not reconstruct or alter patient labels. The current CSV uses the author-corrected independent pre-consensus ratings from `axSpA_konsensus.xlsx`: 740/824 paired ratings agree (89.8%), with unweighted Cohen kappa 0.855 (95% CI 0.825–0.884). The corrected source changes 89 individual reader entries and no consensus labels; model training and evaluation results are unchanged.

The manuscript displays the primary validation-derived binary threshold as 0.271, rounded for presentation. The full-precision value used in all computations is retained in `analysis/yolo11l-cls/seed_0/validation_selected_threshold.json`.

Prediction rows include a technical image identifier, relative path, source-file SHA-256, reference/predicted classes and the four class probabilities. Matching NPZs contain only identifiers, hashes, labels, probabilities and class names; they contain no pixels or Python-object arrays. Probabilities are retained at their exported precision. See [`data/README.md`](data/README.md) for field definitions and uncertainty conventions.

Recreate the statistical figures and cohort-flow diagram without medical images:

```bash
python code/build_publication_figures.py --source . --output reproduced_figures
```

This generates Figures 1–4 and S1–S3. Figure 5 requires the separately held audited CAM image inputs and is enabled only with `--include-private-cam`; it is omitted by default. The plotting adapter changes input/output paths, adds this explicit image-access option, and corrects the study-design wording in the Figure 1 and S2 captions. Numerical plotting methods and data are unchanged. Its source hashes and reversible diff from the executed publication script are recorded in `provenance/public_plot_adapter.*`. The coordinated manuscript correction was completed on 2026-10-06. The adapter's original status record is retained as historical provenance; current publication status is recorded in `release_status.json`.

## Reproduce training or inference

The recorded environment was Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.163 and an NVIDIA A100-SXM4-80GB. Core versions are pinned. The full Colab package list is an observation record, not a portable installation recipe.

```bash
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For a new training reproduction, obtain the accepted image archive and use the notebook, changing its `PROJECT` directory to a **new empty directory** while retaining all scientific settings. Running all cells starts actual GPU training. The notebook checks input hashes, full-batch GPU feasibility, the 20-run plan and completion before external inference. Do not remove an existing evaluation lock to continue fitting. A new run is a reproduction attempt, not recovery of the exact original checkpoint bytes. Future reruns use a cohort whose original evaluation has already been completed; they do not create another untouched external validation. `study_design.json` distinguishes this future-reproduction scope from the author-confirmed original study.

The preparation script accepts three specifically registered archive checksums: the original 827-file archive, its previously verified 824-file version, and `Baza docelowa.zip` supplied on 2026-10-25. Every one of the latter archive's 824 images matches the frozen RGB pixels, dimensions, paths, labels and split assignments, but its PNG file bytes differ. For this registered representation, preparation verifies the archive and reference checksums and all RGB hashes, then records the actual PNG byte hashes in a separate reproduction manifest. Only a new `future_reproduction` plan may use that manifest. Original-study records, published predictions and the authoritative historical manifest remain unchanged. See [`provenance/png_byte_variant_registration_2026-10-25.json`](provenance/png_byte_variant_registration_2026-10-25.json). Archive filenames may differ; acceptance depends on the exact registered checksum.

To repeat inference with the original selected checkpoints, restore the `runs/MODEL/seed_N/selected.pt` files from the separate archives and verify them against `checkpoints/manifest.json`. Retain the original run metadata supplied here. The active evaluation script recognizes the documented correction through the hash-bound ledger; it does not rewrite the original plan or result records. Use the private dataset and a **new output directory**:

```bash
python code/evaluate_models.py \
  --data-root /path/to/prepared_dataset \
  --training-root runs \
  --manifest data/clean_manifest.csv \
  --output-dir reproduced_analysis \
  --device cuda:0 --batch-size 32 --bootstrap-replicates 2000
```

The dataset root must contain `train/`, `val/` and `test/` with the fixed class folders. Standard Ultralytics classification transforms must not replace the supplied `SpineResizePad` transforms. Keep `code/spine_transforms.py` importable when loading checkpoints. In a new reproduction project, the notebook's Eigen-CAM replay verifies the 150 primary external probability vectors against that project's own completed evaluation before generating panels. To replay the archived original predictions, use the original checkpoints and direct command below.

With the original selected checkpoint and prepared PNGs restored, the same replay can be invoked directly against the supplied locked evaluation exports:

```bash
python code/eigen_cam_replay.py \
  --dataset-root /path/to/prepared_dataset \
  --training-root runs --evaluation-root analysis \
  --frozen-code-root provenance/original_code --output-root private_cam_replay \
  --device cuda:0 --batch-size 32
```

Use a new private output directory so that recorded CAM metadata are preserved. `--validate-only` checks frozen inputs without a model forward pass. The saved ledger at [`eigen_cam/eigen_cam_panels.csv`](eigen_cam/eigen_cam_panels.csv) links panels a–h to source paths, identifiers and hashes. The separate Figure 5 layout option requires the original private panel input/overlay/array files at the relative locations listed in that ledger; numeric prediction NPZs and private CAM image/feature NPZs are different artifacts.

On 2026-10-25, all eight source PNGs were located in the author's subsequently supplied archive. Their decoded RGB hashes and reproduced model-input tensor hashes exactly match the frozen references. File-byte hashes differ; the reason for the file-byte differences has not been established. The separate metadata-only audit, `eigen_cam/source_png_verification_2026-10-25.json`, retains both historical and current hashes without rewriting the original ledger. The subsequent full-cohort check confirmed identical RGB content, dimensions and assignments for all 824 images; its separate record is `provenance/png_byte_variant_verification_2026-10-25.json`. The new file representation is registered for future reproduction as described above. Locked replay of the original checkpoints against the archived outputs continues to require the original PNG bytes and historical manifest; its byte checks are not disabled.

## Interpretation and reproducibility limits

- Reference labels describe radiographic morphology, not a clinical diagnosis of axSpA. The binary grouping of parasyndesmophytes/syndesmophytes versus normal/osteophytes is also a morphology endpoint.
- The classes are nominal. Quadratic weighted κ, within-one accuracy and distances between class numbers are not used.
- Shared input geometry does not make the training pipelines identical. Normalization, optimizer, EMA, batch size and epoch budget differ. The recorded YOLO11 and YOLO26 classification backbones/heads have the same layer layout, with distinct pretrained initialization hashes; their names alone do not demonstrate an architectural difference.
- Uncertainty intervals for a fixed checkpoint differ from variability across training seeds. Bootstrap analyses use 2,000 image-level replicates for seed 0 of each configuration; exact binomial intervals are used for proportions. See each metric JSON for its actual bootstrap seed and valid-replicate count.
- Eigen-CAM is a class-agnostic qualitative visualization. It does not prove lesion localization or causal model reasoning.
- Available age/sex/segment data are aggregate cohort summaries; no patient-level linkage was available for demographic subgroup performance estimates.
- The manual, content-independent development split is not a random sample and does not guarantee equivalent train/validation case mix. Filename numbering alone cannot resolve this limitation. The externally held-out cohort provides a separate assessment, subject to its own sample size and population limits.
- Training enabled deterministic algorithms and disabled cuDNN benchmarking and TF32 for matrix multiplication and cuDNN. These flags were not separately recorded for the locked evaluation. The subsequent CAM replay recorded deterministic algorithms and cuDNN deterministic mode disabled, cuDNN TF32 enabled and matrix-multiplication TF32 disabled; all 150 primary external probability vectors nevertheless matched exactly. Bitwise reproducibility across environments is not guaranteed.

The unchanged preparation code retains its required 827-entry input manifest to recognize the earlier source archive and remove three documented duplicate training copies. This provenance input is **not** the denominator for the current study; all reported results use the accepted 824-image manifest. The original frozen protocol also records the initial CAM script hash. The actual reported panels were produced by the replay source preserved at `provenance/original_code/eigen_cam_replay.py` (SHA-256 `7a0b9deb225e36c8b426f6fc2dbf6cd993187733783db3625c8ec44a37a1020c`); `code/eigen_cam_replay.py` is its active successor with the documented study-design metadata correction.

## Study-design description correction

The author's **2026-10-06** clarification supersedes the earlier description of an unknown or random allocation and an already examined external cohort. The original training/evaluation code had hardcoded `external_test_previously_examined=True`; that field was descriptive metadata, not an empirical measurement of prior cohort use. [`study_design.json`](study_design.json) records `original_study.external_test_previously_examined=false` and the manual allocation method. Its separate `future_reproduction` section acknowledges that any subsequent rerun occurs after the completed original evaluation. [`provenance/study_design_correction.json`](provenance/study_design_correction.json) binds the correction to the unchanged original records and preserved source code. Original frozen plans, locks, completion records, predictions, selected weights and numerical results are not rewritten. Active scripts and notebook payloads use the corrected description with its explicit scope. This technical correction involved no retraining, new inference, reassignment of images or recalculation of results.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) for the software citation. Repository code retains the existing [`MIT license`](LICENSE). Third-party dependencies and pretrained model assets retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
