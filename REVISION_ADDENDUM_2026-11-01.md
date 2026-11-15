# Cohort methods and additional analyses

This guide accompanies **materials version 2.1.0**, incorporating the author-confirmed cohort description and the three additional analyses reported in the current manuscript and [supplement](publication/supplementary_methods_results.pdf). The filename retains the date of the underlying methods clarification. Publication metadata are maintained in `CITATION.cff` and `release_status.json`; original execution dates and frozen analysis records retain their original meaning.

## Cohort definition and screening

The cohort comprised patients with chronic back pain referred for spinal radiography by a rheumatologist. Clinical suspicion of axSpA was not required to define the cohort. M.P. reviewed the radiographs to exclude DISH, fractures and evidence of previous spinal surgery. Clinical history beyond the information that patients had chronic back pain was unavailable. Severe degenerative changes were not excluded.

| Screening count | Łódź | Końskie | Total |
|---|---:|---:|---:|
| Before exclusions, reconstructed | 716 | 163 | 879 |
| DISH | 8 | 2 | 10 |
| Fractures | 13 | 3 | 16 |
| Previous spinal surgery | 12 | 3 | 15 |
| Artifacts or poor image quality | 9 | 5 | 14 |
| All exclusions | 42 | 13 | 55 |
| Included | 674 | 150 | 824 |

The author confirmed that technical exclusions were additional to the three radiographic categories. Initial totals are arithmetic reconstructions from these confirmed exclusions and final cohort sizes. This accounting does not assert an order between radiographic eligibility review and quality review. The accepted memberships remain 579 training, 95 internal-validation and 150 external-test images. The structured, source-attributed record is `cohort_clarification_2026-11-01.json`.

These facts supersede earlier statements requiring suspected axSpA, describing history-based screening, or saying that no DISH, fracture or postoperative cases were present before screening. Frozen execution records and historical correction ledgers are preserved. The current Figure S1 generation uses the corrected counts; other numerical plotting data and code are unchanged.

## Post hoc additional analyses

`additional_analyses/` now contains three separately labeled **post hoc exploratory analyses**: geometry/brightness and reader-versus-own-consensus analyses completed on 25 October 2026, and content ablation completed on 1 November 2026. All were performed after the primary experiment results were known; none is part of the frozen original training plan or study preregistration. Original analysis artifact bytes are preserved. None retrained a CNN or changed the original predictions, labels or split assignments; the third generated separately labeled predictions for artificial ablated inputs.

* **Geometry and brightness:** a fixed multinomial logistic regression used prepared-PNG width, height, width/height and mean RGB brightness. Standardization and fitting used the 579 training cases only, without hyperparameter tuning on validation or test. External accuracy was 67/150 (44.7%; Wilson 95% CI 36.9–52.7), compared descriptively with 55/150 (36.7%) for the training-majority-class baseline. The difference was not subjected to a significance test. Kruskal–Wallis tests and Holm adjustment cover all four features separately within each split. Association of these features with labels neither proves their use by YOLO nor excludes shortcut learning. The folder includes all feature-level data, per-class descriptives, tests, predictions, coefficients, recorded environment and source code. A plan was saved before this additional computation, after the main results were available; its filename must not be read as a claim of prospective study preregistration.
* **Initial reader versus own consensus:** M.P. agreed with consensus in 142/150 (94.7%, κ 0.924), and P.G. in 143/150 (95.3%, κ 0.934). Both readers co-created the reference; this is dependent descriptive agreement, not an independent human benchmark or evidence of model–reader equivalence. No superiority, noninferiority or significance tests against model performance were performed. The corrected reader-rating CSV and consensus labels are unchanged.

* **Frozen-checkpoint content ablation:** the original 150-image probability replay matched exactly (maximum difference 0); the gate required a difference no greater than 0.0001. Fixed-gray and per-image-mean-gray PNGs retained dimensions, aspect ratio and padding but erased all spatial content. Both produced 55/150 correct (36.7%; Wilson 95% CI 29.4–44.6), balanced accuracy 25%, macro-F1 0.134 and κ 0, with all 150 predictions normal. Mean maximum probabilities remained 94.52% and 93.46%. The paired original-minus-ablated accuracy difference was 55.33 percentage points (10,000-replicate bootstrap 95% CI 46.67–64.00) in either condition. Under these conditions, the retained padding footprint alone was insufficient to reproduce the original class assignments. These controls do not establish reliance specifically on bone features. This majority-class collapse on out-of-distribution inputs does not isolate geometry's contribution on intact images or exclude shortcuts. The plan was frozen before this additional inference, after primary and geometry findings were known. A prior CPU replay failed the unchanged tolerance and stopped before ablations; the subsequent GPU run passed. The original failure and successful run are both retained.

The same two radiologists labeled the development and external cohorts. External evaluation concerns a separate patient cohort and acquisition site; transfer to an independently established reader reference was not evaluated. Geometry associations and the new content-ablation results do not remove this reference-standard limitation.

The primary model result remains 138/150 (92.0%) for the designated seed-0 run; the five-seed mean remains 88.4%. The reference run happened to be the highest-scoring YOLO11 seed externally. Repeated seeds use the same cohort and are not independent external validations.

## Reproduction of the additional calculations

Run from the repository root and use **new output directories**; do not overwrite the included analysis records. The geometry script requires the original, accepted private 824-PNG ZIP (SHA-256 `ba4c5ece78779f67b209f839cec75f70aa480c07bc8ee1dee008f4594a96613d`) and the frozen manifest. The included execution records report the package versions. Source PNGs are available from the corresponding author on reasonable request.

```bash
mkdir reproduced_geometry
cp additional_analyses/geometry_brightness/analysis_plan_frozen_before_execution.json reproduced_geometry/
python additional_analyses/geometry_brightness/analyze_geometry_brightness.py \
  /path/to/original_824_png.zip data/clean_manifest.csv reproduced_geometry
python additional_analyses/reader_vs_own_consensus/compute_reader_consensus_agreement.py \
  --input data/reader_ratings.csv --output-dir reproduced_reader_consensus
```

The included historical plan remains a record of when the prior analysis was specified. Copying it for reproduction does not imply that a new plan has been preregistered. New output timestamps and local source paths may differ; compare numerical outputs and hashes of underlying inputs.

For the new content-ablation analysis, use [`additional_analyses/shape_sensitivity/README.md`](additional_analyses/shape_sensitivity/README.md). It documents the actual frozen script interface, exact `reference_source/` directory layout, GPU flags, required private input identities and unchanged replay gate. Do not substitute active code for frozen source files or overwrite historical outputs. All copied analysis artifacts are listed by SHA-256 in that folder's `source_artifact_manifest.json`.

## Unchanged experimental evidence

Original training plans, run records, locked evaluation, prediction tables and NPZs, all threshold JSONs, image and reader manifests, selected-checkpoint hashes and Eigen-CAM provenance are preserved byte for byte. The release preserves the original recorded outputs and their input identities. This update changes cohort documentation, the Figure S1 plotting block and adds separately labeled post hoc analyses; it does not establish a new untouched external evaluation.

This addendum includes the content-ablation materials and the author-confirmed final cohort description: 674 development images (579 training, 95 validation) and 150 external images. Current documentation and Figure S1 descriptive captions use those counts; the clinical eligibility flow remains 879 assessed and 55 excluded. Original scientific code, locked results, manifests, checkpoint hashes and frozen provenance are preserved. These current descriptions and additional analyses are included in the version 2.1.0 software package.
