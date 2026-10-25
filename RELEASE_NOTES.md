# Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model

This is the completed cleaned-data experiment accompanying the current Diagnostics manuscript. The current GitHub release is v2.0.0. Code and non-image reproducibility materials are archived at [DOI: 10.5281/zenodo.23649843](https://doi.org/10.5281/zenodo.23649843).

- 824 images: 579 training and 95 validation from Łódź, with 150 external images from Końskie.
- Primary model: YOLO11l-cls seed 0, 138/150 correct external classifications (92.0%). Four configurations, five seeds each; all exported validation/test predictions and recorded settings are included.
- Manual within-class allocation of numerically ordered filename ranges, fixed before training without content-based or performance-based selection. The external cohort was reserved for final evaluation, as confirmed by the author.
- Windows Snipping Tool preparation before labeling; whole prepared PNG computational input, proportional resize and black padding.
- Active code, reproduction notebook, manifests, metrics, original executed code and hash-bound correction records constitute the current repository tree.
- The eight Figure 5 source images have matching RGB-pixel and model-input hashes. The new verification contains metadata only. Source PNGs and model weights remain separate research materials, available from the corresponding author as described in README.
- All 824 images in the subsequently supplied PNG archive match the frozen RGB pixels and assignments. This exact archive is registered for future reproduction with its own actual-file-hash manifest; the original-study manifest and locked replay checks remain unchanged.
- Active preparation metadata now states the confirmed source-preparation method. Notebook completion-check descriptions match the implemented checks. These technical corrections do not change training methods or reported results.

The author supplied corrected independent pre-consensus reader ratings, replacing erroneous initial reader data. Agreement is 740/824 (89.8%), unweighted Cohen kappa 0.855 (95% CI 0.825–0.884), with 84 disagreements. All 824 consensus labels remain unchanged. No training or new inference was performed; all model results and dataset assignments are unchanged. The DOI above identifies these current computational materials; source PNGs and checkpoint binaries are not included in this deposit.
