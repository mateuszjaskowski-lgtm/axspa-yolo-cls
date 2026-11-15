# Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model

Companion materials for **Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model**, by Mateusz Jaśkowski, Paweł Górski, Zbigniew Guzera and Michał Podgórski, prepared for *Diagnostics*.

**Materials version 2.1.0 (15 November 2026).** This package incorporates the current manuscript's cohort description, computational methods and all three additional analyses (Supplementary Tables S7–S9). See [RELEASE_NOTES.md](RELEASE_NOTES.md), [supplementary methods and results](publication/supplementary_methods_results.pdf) and [the methods and additional-analysis guide](REVISION_ADDENDUM_2026-11-01.md). The [15 November revision summary](REVISION_ADDENDUM_2026-11-15.md) records the current methods, ethics and figure updates. The public supplementary PDF contains the methods, tables and Figures S1–S3. The Figure S4 source ledger and reconstruction code are in `figure_revision/`. The Zenodo [all-versions DOI](https://doi.org/10.5281/zenodo.22850851) identifies the deposit series; use the version-specific DOI displayed on the corresponding Zenodo record when citing a particular archived package.

Reproducibility materials for a retrospective two-center study of **four nominal morphology classes** on prepared lateral spinal radiographs. The reference model is **YOLO11l-cls, seed 0**. YOLO26l-cls, ResNet50 and EfficientNet-B0 are comparison configurations, each evaluated with seeds 0–4.

This repository describes the completed fixed-cohort experiment. The external cohort was reserved for the planned final evaluation after training and checkpoint selection and was not used for model development. The study is retrospective; prospective clinical utility has not been established. The original analysis plan was agreed verbally, without prospective registration; the archived computational plan for the fixed-cohort experiment is a separate record.

## Data and labels

The cohort comprised adults with chronic back pain referred for spinal radiography by a rheumatologist, with radiographs acquired from January 2022 through December 2025. Clinical suspicion of axSpA was not an inclusion requirement. M.P. performed a preliminary radiographic review for DISH, fractures and evidence of previous spinal surgery. No additional clinical history was available beyond chronic back pain. The author-confirmed exclusion counts are 33 radiographic plus 9 technical exclusions in Łódź and 8 plus 5 in Końskie. Thus the reconstructed unique-case flow is 716 to 674 and 163 to 150, respectively (879 minus 55 equals 824). Severe degeneration was not excluded. See [`cohort_clarification_2026-11-01.json`](cohort_clarification_2026-11-01.json) for the per-reason counts and source attribution.

| Split | Normal | Osteophytes | Parasyndesmophytes | Syndesmophytes | Total |
|---|---:|---:|---:|---:|---:|
| Training, Łódź | 211 | 210 | 81 | 77 | 579 |
| Validation, Łódź | 37 | 34 | 12 | 12 | 95 |
| External test, Końskie | 55 | 54 | 20 | 21 | 150 |

The accepted dataset contains **824 images**. The development split was made manually in Windows Explorer: within each class folder, PNGs were sorted numerically and contiguous ranges were moved to validation until approximately 15% of that class had been selected. The blocks were selected without regard to image content, before the first training, and assignments were not changed in response to model outcomes. This allocation was nonrandom, and acquisition chronology was not established from the numerical filenames. No `scikit-learn` split or random-allocation seed was used. Validation contains **95/674 development images (14.1%)**; class-specific fractions are given in [`data/README.md`](data/README.md). The authoritative memberships and image hashes are in [`data/clean_manifest.csv`](data/clean_manifest.csv). Checks of file bytes and decoded RGB pixels confirmed exact-image uniqueness. Each patient contributed one image.

Class order is fixed: `normal`, `osteofity`, `parasyndesmofity`, `syndesmofity`. Integer labels 0–3 identify categories and do not represent severity grades. The study's syndesmophyte category represents bridging lesions under its reference-labeling protocol.

M.J. prepared PNGs with Windows Snipping Tool, selecting the entire examined spinal segment and its surroundings before radiologist labeling. Both radiologists assessed these same prepared PNGs. Images supplied by both centers had no textual annotations or identifying information. No manual brightness/contrast adjustment, CLAHE or other image enhancement was applied; preceding export/display settings were not documented. Computational preprocessing preserves the complete prepared PNG field, resizes proportionally with bilinear interpolation, and adds centered black padding to 640 × 640 pixels. Training uses horizontal flips and brightness/contrast jitter. Source preparation and training augmentation are separate steps.

## Current results

The reference YOLO11l-cls seed-0 model classified **138/150 external images correctly: 92.0% accuracy (exact 95% CI 86.4–95.8%)**, with macro-F1 0.9265, unweighted Cohen's κ 0.8862 and macro one-versus-rest AUC 0.9875.

| Configuration | Seed-0 external accuracy | Five-seed mean ± sample SD |
|---|---:|---:|
| YOLO11l-cls, reference | 92.00% | 88.40% ± 3.52 percentage points |
| YOLO26l-cls | 87.33% | 89.47% ± 1.28 percentage points |
| ResNet50 | 93.33% | 91.07% ± 3.35 percentage points |
| EfficientNet-B0 | 90.67% | 89.60% ± 1.12 percentage points |

The archived plan designated YOLO11l-cls seed 0 as primary before external evaluation. It happened to have the highest external YOLO11 accuracy among the five runs; the five-seed mean was 88.4%, with a range of 84.0–92.0%. The reference run was not chosen by its external accuracy. Five seeds measure training variability on the **same 150 images** and do not increase the number of independent patients. The three seed-0 accuracy comparisons against YOLO11 were not significant after Holm adjustment; this does not establish equivalence or imply that every secondary metric has a null difference. Detailed results are in [`analysis/`](analysis/).

## Additional exploratory analyses

All three analyses were performed after the primary results were known and are reported as post hoc exploratory analyses, not as prospective study preregistration. Their saved results are included in this release.

- **Geometry and brightness (Table S7):** a logistic regression fitted only on the 579 training images used width, height, aspect ratio and mean RGB brightness. External accuracy was 67/150 (44.7%; Wilson 95% CI 36.9–52.7), compared descriptively with the training-majority baseline of 55/150 (36.7%). Image height and aspect ratio differed across external reference classes. These associations do not establish that YOLO used these characteristics or exclude shortcut learning.
- **Initial readers versus their own consensus (Table S8):** M.P. agreed in 142/150 external images (94.7%; κ 0.924) and P.G. in 143/150 (95.3%; κ 0.934). Both readers co-created the consensus. This dependent comparison does not establish model–reader equivalence or provide an independent human benchmark.
- **Frozen-checkpoint content ablation (Table S9):** the two gray-content controls test loss of discrimination when spatial radiographic information is removed.

The inference-only sensitivity test required identical original decisions and a maximum absolute probability difference no greater than 0.0001. It reproduced all 150 original primary-model probability vectors exactly (maximum difference 0) before either ablation. Replacing the entire PNG content with fixed gray 128 or each image's mean gray, while preserving aspect ratio and the resize/padding pathway, gave 55/150 correct classifications (36.7%) in either condition: **all predictions were normal**, balanced accuracy was 25%, macro-F1 0.134 and unweighted κ 0. Mean maximum probabilities remained 94.52% and 93.46%, respectively. Under these control conditions, the retained padding footprint alone was insufficient to reproduce the original class assignments. These controls do not establish reliance specifically on bone features. They are out-of-distribution inputs and neither isolate interactions between geometry and radiographic content on intact images nor exclude shortcut learning. Methods, full predictions and preserved execution records are in [`additional_analyses/shape_sensitivity/`](additional_analyses/shape_sensitivity/).

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
- `study_design.json`: author-confirmed allocation and original external-cohort-use description; the later cohort eligibility clarification is in `cohort_clarification_2026-11-01.json`.
- `additional_analyses/`: three separately labeled post hoc analyses: geometry/brightness, initial readers versus their co-created consensus, and frozen-checkpoint content ablation. They were performed after the primary results were known; see the revision addendum and `additional_analyses/shape_sensitivity/README.md`. Their outputs do not replace the original evaluation.
- `provenance/`: original executed assets in `original_code/`, exact-source hashes, technical-repair records, notebook code-integrity records and the hash-bound `study_design_correction.json` correction ledger.
- `environment/`: observed environment and full recorded Colab package listing.
- `publication/supplementary_methods_results.pdf`: current Supplementary Methods S1–S3, Tables S1–S9 and Figures S1–S3. The manuscript and administrative submission documents are not part of this software deposit.

The repository contains **no source PNGs, checkpoint binaries or CAM image pixels**. PNG images are available from the corresponding author, [Mateusz Jaśkowski](mailto:mateuszjaskowski@gmail.com), on reasonable request. Checkpoints are separate research artifacts; use the checkpoint manifest to identify and verify the exact files, and contact the corresponding author if you do not have the associated archives. The code license does not grant rights to restricted source images or third-party model assets.

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

`data/reader_ratings.csv` contains only row order, center and the two initial labels plus consensus. The accompanying script reproduces the saved agreement estimates and all three 10,000-resample bootstrap intervals; it does not reconstruct or alter patient labels. The current CSV uses the author-corrected independent pre-consensus ratings from `axSpA_konsensus.xlsx`: 740/824 paired ratings agree (89.8%), with unweighted Cohen kappa 0.855 (95% CI 0.825–0.884). All consensus labels and model training/evaluation results are unchanged.

The manuscript displays the primary validation-derived binary threshold as 0.271, rounded for presentation. The full-precision value used in all computations is retained in `analysis/yolo11l-cls/seed_0/validation_selected_threshold.json`.

Prediction rows include a technical image identifier, relative path, source-file SHA-256, reference/predicted classes and the four class probabilities. Matching NPZs contain only identifiers, hashes, labels, probabilities and class names; they contain no pixels or Python-object arrays. Probabilities are retained at their exported precision. See [`data/README.md`](data/README.md) for field definitions and uncertainty conventions.

Recreate the statistical figures and cohort-flow diagram without medical images:

```bash
python code/build_publication_figures.py --source . --output reproduced_figures
```

This generates Figures 1–4 and S1–S3. To reproduce the current versions of Figures 2 and 4, run `python figure_revision/revise_roc_figures.py --source analysis/yolo11l-cls/seed_0 --output reproduced_figures`. Figure 5 requires the separately held audited CAM image inputs and is enabled only with `--include-private-cam`; it is omitted by default. The original plotting adapter changes input/output paths, adds this explicit image-access option, and corrects the study-design wording in the Figure 1 and S2 captions. Its Figure S1 block uses the current author-confirmed cohort flow. Numerical model plotting methods and data are unchanged; the new S1-specific source changes are documented in `provenance/cohort_flow_plot_correction_2026-11-01.json`. Its source hashes and reversible diff from the executed publication script are recorded in `provenance/public_plot_adapter.*`. Dated adapter records retain their original execution meaning. Current package scope and publication status are recorded in `release_status.json`.

## Reproduce training or inference

The recorded environment was Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.163 and an NVIDIA A100-SXM4-80GB. Core versions are pinned. The full Colab package list is an observation record, not a portable installation recipe.

```bash
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For a new training reproduction, obtain the accepted image archive and use the notebook, changing its `PROJECT` directory to a **new empty directory** while retaining all scientific settings. Running all cells starts actual GPU training. The notebook checks input hashes, full-batch GPU feasibility, the 20-run plan and completion before external inference. Do not remove an existing evaluation lock to continue fitting. A new run is a reproduction attempt, not recovery of the exact original checkpoint bytes. Future reruns use a cohort whose original evaluation has already been completed; they do not create another untouched external validation. `study_design.json` distinguishes this future-reproduction scope from the author-confirmed original study.

Active preparation accepts only the registered 824-image archives and verifies their fixed paths, dimensions, hashes and assignments without generating a new split. Its default reference is `code/reference_clean_manifest.csv`; current preparation and notebook changes are documented in `provenance/current_dataset_release_2026-11-08.json`. Use the accepted dataset with its registered archive checksum. The `Baza docelowa.zip` representation supplied on 2026-10-25 contains the same 824 RGB images, dimensions, paths, labels and split assignments as the frozen study manifest, but its PNG file bytes differ. For this registered representation, preparation verifies the archive and reference checksums and all RGB hashes, then records the actual PNG byte hashes in a separate reproduction manifest. Only a new `future_reproduction` plan may use that manifest. Original-study records, published predictions and the authoritative historical manifest remain unchanged. See [`provenance/png_byte_variant_registration_2026-10-25.json`](provenance/png_byte_variant_registration_2026-10-25.json). Archive filenames may differ; acceptance depends on the exact registered checksum.

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
- The same two radiologists labeled both cohorts and resolved their disagreements by consensus. External evaluation therefore concerns a separate patient cohort and acquisition site; transportability to an independently established reader reference was not tested.
- The classes are nominal. Quadratic weighted κ, within-one accuracy and distances between class numbers are not used.
- Shared input geometry does not make the training pipelines identical. Normalization, optimizer, EMA, batch size and epoch budget differ. The stored architecture descriptions of the two YOLO variants shared backbone and head layer definitions, while their pretrained checkpoint hashes differed. Different file hashes alone do not establish differences between weight tensors, and the model names alone do not demonstrate an architectural difference.
- Uncertainty intervals for a fixed checkpoint differ from variability across training seeds. Bootstrap analyses use 2,000 image-level replicates for seed 0 of each configuration; exact binomial intervals are used for proportions. See each metric JSON for its actual bootstrap seed and valid-replicate count.
- Eigen-CAM is a class-agnostic qualitative visualization. It does not prove lesion localization or causal model reasoning.
- Available age/sex/segment data are aggregate cohort summaries; no patient-level linkage was available for demographic subgroup performance estimates.
- The manual, content-independent development split is not a random sample and does not guarantee equivalent train/validation case mix. Acquisition chronology was not established from the numerical filenames. The externally held-out cohort provides a separate assessment, subject to its own sample size and population limits.
- Training enabled deterministic algorithms and disabled cuDNN benchmarking and TF32 for matrix multiplication and cuDNN. These flags were not separately recorded for the locked evaluation. The subsequent CAM replay recorded deterministic algorithms and cuDNN deterministic mode disabled, cuDNN TF32 enabled and matrix-multiplication TF32 disabled; all 150 primary external probability vectors nevertheless matched exactly. Bitwise reproducibility across environments is not guaranteed.

The reported study uses the fixed 824-image manifest: 674 development images (579 training and 95 validation) and 150 external images. Original execution assets remain preserved for computational provenance. The original frozen protocol records the initial CAM script hash. The actual reported panels were produced by the replay source preserved at `provenance/original_code/eigen_cam_replay.py` (SHA-256 `7a0b9deb225e36c8b426f6fc2dbf6cd993187733783db3625c8ec44a37a1020c`); `code/eigen_cam_replay.py` is its active successor with the documented study-design metadata correction.

## Execution provenance and current study description

[`study_design.json`](study_design.json) records `original_study.external_test_previously_examined=false` and the manual allocation method. Its separate `future_reproduction` section acknowledges that any subsequent rerun occurs after the completed original evaluation. [`provenance/study_design_correction.json`](provenance/study_design_correction.json) binds the correction to the unchanged original records and preserved source code. Original frozen plans, locks, completion records, predictions, selected weights and numerical results are not rewritten. Active scripts and notebook payloads use the corrected description with its explicit scope. This technical correction involved no retraining, new inference, reassignment of images or recalculation of results.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) for the software citation. Repository code retains the existing [`MIT license`](LICENSE). Third-party dependencies and pretrained model assets retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
