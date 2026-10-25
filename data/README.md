# Public data dictionary

`clean_manifest.csv` is the authoritative 824-image split manifest. It contains technical file identifiers only, with no patient names, dates of birth or individual demographic variables.

This archived manifest identifies the original PNG file bytes. The subsequently registered `Baza docelowa.zip` representation has identical RGB pixels, dimensions and assignments for all 824 images, but different file-byte hashes. For future reproduction only, `code/prepare_data.py` creates a separate manifest containing those actual hashes and records the historical-to-current mapping in its preparation audit. The archived `data/clean_manifest.csv` and published prediction exports are not replaced. The exact original manifest checksum is `3016eaeeb8af3745c64b159c5189549fadde3753aa08d232ea7e7d450f4e4dfd`; the registered reproduction manifest checksum is `f98cc4992fe3da8140ed1ab033361e86188fc3e9c9c04d814ac09af3bb869642`. Details and all 824 comparisons are in `provenance/png_byte_variant_registration_2026-10-25.json` and `provenance/png_byte_variant_verification_2026-10-25.json` at the repository root.

The author's **2026-10-06** clarification describes manual allocation in Windows Explorer. Within numerically sorted class folders, contiguous file ranges were moved to validation until approximately 15% of each class had been selected. Selection was independent of image content; the folders were fixed before the first training and were not reassigned according to results. No random or `scikit-learn` split was used. File numbering does not establish acquisition chronology.

The retained development memberships, after the documented removal of duplicate training copies, are:

| Class | Training | Validation | Development total | Validation fraction |
|---|---:|---:|---:|---:|
| Normal | 211 | 37 | 248 | 14.9% |
| Osteophytes | 210 | 34 | 244 | 13.9% |
| Parasyndesmophytes | 81 | 12 | 93 | 12.9% |
| Syndesmophytes | 77 | 12 | 89 | 13.5% |
| Total | 579 | 95 | 674 | 14.1% |

The external cohort contains 150 images. The author confirmed that it had not been used before the final evaluation. `../study_design.json` and `../provenance/study_design_correction.json` document this clarification and the correction of the erroneous hardcoded `external_test_previously_examined=True` field in the original workflow. The preserved original metadata are historical execution records; the correction does not change any image memberships, labels, predictions or numerical results.

| Column | Meaning |
|---|---|
| `split` | `train`, `val` or `test` |
| `class_name` | Original class-folder name |
| `label` | Nominal integer class index 0–3 |
| `relative_path` | Technical PNG path within the private prepared dataset |
| `image_id` | Stable technical identifier used to join prediction exports |
| `sha256` | Hash of the exact prepared PNG file bytes |
| `pixel_sha256`, `independent_rgb_sha256` | Pixel-content hashes recorded by the data audit |
| `width`, `height` | Prepared PNG dimensions, before model resizing/padding |

The 40 `analysis/MODEL/seed_N/{val,test}_predictions.csv` files contain `image_id`, `relative_path`, `sha256`, `true_label`, `true_class`, `predicted_label`, `predicted_class`, `p_normal`, `p_osteofity`, `p_parasyndesmofity` and `p_syndesmofity`. The four probabilities are exported model outputs, not calibrated clinical disease probabilities. All seeds use the same split and reference labels.

The 40 matching `*_predictions.npz` files contain exactly five arrays: `image_id`, `sha256`, `y_true`, `probabilities` and `class_names`. Identifiers and class names are fixed-width strings, labels are integers, and probabilities are numeric arrays of shape `(95, 4)` or `(150, 4)`. No pixels, feature maps or object-dtype arrays are included. Values and ordering agree exactly with the accompanying CSVs. These files are required by the CAM provenance checks.

The neighboring `*_metrics.json` files provide confusion matrices with **true classes as rows and predicted classes as columns**, fixed class order, counts, point estimates and available uncertainty intervals. Bootstrap intervals are computed for seed 0 of every configuration; other seeds have point estimates and exact binomial intervals for proportions. Each JSON records the method and actual bootstrap seed. Five-seed means and sample standard deviations are in `analysis/five_seed_summary.csv`.

`validation_selected_threshold.json` records the threshold chosen on validation data for the binary morphology grouping, not a clinical diagnosis. `analysis/paired_comparisons_seed0.json` includes the three paired reference-versus-comparator accuracy tests, Holm-adjusted p-values and pointwise bootstrap differences for secondary metrics. Those pointwise intervals are not simultaneous multiplicity-adjusted confidence intervals.

These files contain no medical image pixels. Prepared PNGs are available from the corresponding author on reasonable request.

`eigen_cam/eigen_cam_panels.csv` maps each of the eight Figure 5 panels to its technical source path, image identifier, source hash, labels, prediction, probabilities and private output-file names. The adjacent JSONs record selection, preprocessing, target layer, runtime flags, probability replay and original artifact hashes. They contain metadata only; referenced input/overlay pixels and activation/map NPZs are deliberately separate from this public Git tree.
