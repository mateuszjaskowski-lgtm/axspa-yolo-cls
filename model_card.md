# Model card

**Task:** image-level classification of the dominant radiographic morphology on prepared lateral spinal PNGs into normal, osteophytes, parasyndesmophytes or syndesmophytes. These are nominal categories, not axSpA severity grades.

**Reference checkpoint:** YOLO11l-cls, seed 0; SHA-256 `2ef70f6c73c43970cbaebed2884e71b1cd374b1545d5239abe6f2243e75c5ac6`. The selected epoch was 63 of 240, based on maximum validation top-1 accuracy with the earliest epoch breaking ties. The 95-image validation accuracy was 88/95 (92.63%).

**Development and evaluation data:** 579 training and 95 validation images from Łódź; 150 external images from Końskie. The source PNGs were manually prepared before radiologist labeling. Input geometry retains the full prepared PNG via proportional bilinear resizing and black padding to 640 × 640. No additional automatic crop is applied. YOLO scales RGB to 0–1; torchvision comparators additionally use their recorded ImageNet normalization.

**Development allocation:** according to the author's clarification of 2026-10-06, numerically sorted class folders were partitioned manually in Windows Explorer by moving contiguous file ranges to validation, targeting approximately 15% per class. Selection did not depend on image content; assignments were fixed before the first training and were not changed according to results. After duplicate-copy removal, the validation fraction is 95/674 (14.1%). This was not a random or `scikit-learn` allocation. Numerical order does not establish acquisition chronology, and content-independent block selection does not guarantee representative train/validation case mix.

**Reference external performance:** accuracy 138/150 (92.0%; exact 95% CI 86.44–95.80%), macro-F1 0.9265, balanced accuracy 0.9369, unweighted Cohen's κ 0.8862 and macro OvR AUC 0.9875. Full estimates and intervals are in `analysis/yolo11l-cls/seed_0/test_metrics.json`.

**Training variability:** YOLO11 external accuracy across seeds 0–4 was 92.00%, 84.00%, 85.33%, 90.67% and 90.00%; mean 88.40%, sample SD 3.52 percentage points. These are repeated fits on the same split, not five independent external cohorts or an ensemble.

**Selection and comparisons:** the primary configuration and seed were fixed before this revision evaluation. All 20 fits were completed and checkpoint hashes frozen before evaluation. Comparators are YOLO26l-cls, ResNet50 and EfficientNet-B0, with five seeds each. The author confirmed on 2026-10-06 that the external cohort had not been used before the final evaluation. Comparisons concern complete training configurations; YOLO11 and YOLO26 classification metadata show the same layer layout with different recorded pretrained weight hashes. The original plan was verbal and was not prospectively registered; the locked computational plan is a separate record for the cleaned-data experiment.

**Reference labels:** dominant morphology was determined by affected vertebral counts, with ties resolved by counting individual outgrowths. The study syndesmophyte class used bridging lesions. These operational labels must not be reinterpreted as a universal definition of syndesmophytes or as a clinical diagnosis.

**Binary endpoint:** summing the parasyndesmophyte and syndesmophyte probabilities defines a morphology grouping. The operating threshold was selected on validation data and applied to the external test. It is not a threshold for diagnosing axSpA. Confidence intervals conditional on the selected threshold do not incorporate threshold-selection uncertainty.

**Explainability:** Eigen-CAM uses the primary checkpoint and batch-consistent FP32 prediction replay. All 150 stored external probability vectors matched during replay. The maps are qualitative and class-agnostic, with no lesion-localization ground truth.

**Numerical reproducibility:** deterministic algorithms, disabled cuDNN benchmarking and disabled matrix-multiplication/cuDNN TF32 were recorded for **training**. The runtime flags of the locked evaluation were not separately recorded. CAM replay recorded `deterministic_algorithms=False`, `cudnn_deterministic=False`, `cudnn_allow_tf32=True` and matrix-multiplication TF32 disabled. Despite these differences, the replay reproduced all 150 stored primary external probability vectors exactly. The recorded training flags must not be attributed to every later computation, and cross-environment bitwise reproducibility is not guaranteed.

**Use:** research reproducibility and methodological comparison on the defined data. Prospective clinical utility, screening performance and generalization to unprepared full examinations, other modalities or different populations have not been established. This software must not replace clinical assessment.

**Availability:** code, frozen settings, nonidentifying technical split records and exported predictions are provided here. PNGs are available on reasonable request to `mjaskowski@zoz.konskie.pl`. Checkpoint binaries are separate from the Git tree; their exact hashes, sizes and archive membership are in `checkpoints/manifest.json`.

**Verified PNG file representation:** the subsequently supplied 824-image archive has identical decoded RGB pixels, dimensions and fixed assignments, with different PNG file bytes. The registered representation is supported for new reproduction projects with its own actual-file-hash manifest. Original-study manifest and prediction hashes are unchanged; locked original-checkpoint replay continues to use the original byte representation. The registration and full comparison are recorded in `provenance/png_byte_variant_registration_2026-10-25.json` and `provenance/png_byte_variant_verification_2026-10-25.json`.

**Descriptive correction:** `study_design.json` and `provenance/study_design_correction.json` document the 2026-10-06 author clarification and correction of the erroneous hardcoded `external_test_previously_examined=True` metadata. Original executed files and hash-bound records remain available; active scripts use the corrected description. No split memberships, training results, predictions or weights changed in this correction.
