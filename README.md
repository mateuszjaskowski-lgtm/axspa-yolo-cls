# Lateral spinal radiograph morphology classification

Research code accompanying **Deep Learning Classification of Morphological Patterns on Lateral Spinal Radiographs in Patients with Suspected Axial Spondyloarthritis**, by Mateusz Jaśkowski, Paweł Górski, Zbigniew Guzera and Michał Podgórski. Manuscript prepared for Diagnostics; no publication or acceptance is claimed.

## Task

Four **nominal morphological categories**, not an ordinal severity scale:

| Code | Category | Operational definition |
|---|---|---|
| 0 | Normal | Absence of assessed structural lesions |
| 1 | Osteophytes | Horizontally oriented vertebral-margin outgrowths |
| 2 | Parasyndesmophytes | Distinct intermediate/non-marginal pattern assigned by readers |
| 3 | Syndesmophytes | Vertically oriented bridging lesions in this study |

Dominant type was determined by affected-vertebra counts; ties by individual outgrowths. Two independent radiologists assigned labels; all 174 disagreements were resolved by consensus. Pre-consensus nominal Cohen's kappa was 0.703 (95% CI 0.663–0.741), exact agreement 653/827.

Secondary binary grouping combines categories 2 and 3. It describes dominant morphology, not clinical axSpA diagnosis or absence of every lesion on mixed images.

## Data and model

January 2022–December 2025: Lodz 686 assessed, 9 quality exclusions, 677 included (582 train, 95 validation); Konskie 155 assessed, 5 quality exclusions, 150 external test. One image per patient from the most painful spinal segment. No included image was omitted from model evaluation. No cases of DISH, prior spinal surgery/instrumented fusion, or vertebral fractures unrelated to axSpA were present in the assessed pools of 686 Lodz and 155 Konskie examinations. Only the 9 and 5 image-quality exclusions occurred in those pools; no counts are inferred for earlier screening.

Complete anonymized PNG files as received were supplied to the model pipeline without manual cropping or additional author-performed preprocessing. Automatic framework transforms are described separately in the manuscript. Device details and separate train/validation demographic summaries are unavailable. Analysis plan agreed verbally; no written plan available for review.

Primary model: ImageNet-pretrained YOLO11l-cls, 640 pixels, batch 100, 240 epochs, seed 0, Ultralytics 8.3.246. Checkpoint history identifies best validation top-1 at epoch 199. Original Python/PyTorch/CUDA versions, GPU and timing measurements are not verified. See docs/training_config.yaml.

## Corrected primary results

| Metric | Internal n=95 | External n=150 |
|---|---:|---:|
| Accuracy | 90/95 = 0.947 | 135/150 = 0.900 |
| Nominal Cohen's kappa | 0.923 | 0.856 |
| Macro one-vs-rest AUC | 0.981891 | 0.987043 |
| Binary dominant-category AUC | 0.996479 | 0.989931 |
| Mean one-vs-rest Brier | 0.026179 | 0.038277 |

External accuracy exact 95% CI: 0.840–0.943. External binary AUC bootstrap CI: 0.977–0.999. Full estimates and methods: audit/.
Internal-derived threshold 0.584019549792572 gives external TP=35, FN=6, FP=2, TN=107. Externally optimized thresholds are exploratory.

Revision comparisons: ResNet50 140/150, EfficientNet-B0 137/150, new YOLO26l-cls 138/150. New archived YOLO26 replaces the earlier incompletely documented 140/150 run: 12,839,748 parameters, best validation epoch 150, Ultralytics 8.4.163, external macro-AUC 0.987679 and binary AUC 0.988812. External data had already been examined; revision comparisons are exploratory. Different pipelines prevent attributing differences solely to architecture.

## Reproduction

Install analysis dependencies with: python -m pip install -r requirements.txt

Run: python scripts/export_metrics.py --published-matrices --output confusion_matrix_audit.json

This reanalyzes manuscript hard-label matrices. It cannot reconstruct probabilities, patient IDs, AUC or prediction provenance. Exported matrices use true rows and predicted columns; original Figure 1 uses the transpose.

With authorized original files:

    python scripts/export_metrics.py --input external_probs.npz --positive-indices 2 3 --threshold 0.584019549792572 --output external_audit.json
    python scripts/audit_reader_agreement.py --input reader_agreement_axspa.xlsx --output reader_audit.json

NPZ schema: Unicode class_names, integer true, float probs (N × 4); a single pair of suffixed _true/_probs keys is also accepted. Preserve canonical class order. Reader XLSX columns: Case_ID, Cohort, M.P., P.G., Consensus; labels 0–3.
Notebooks run these audited functions. Case-level ratings/probabilities, original radiographs and trained weights are not distributed here; access is subject to institutional permission. Included JSONs contain aggregate results.

## Inference and future evaluations

    python -m pip install -r requirements-inference.txt
    python scripts/inference.py --model best.pt --image radiograph.png
    python scripts/evaluate_folder.py --model best.pt --data dataset/test --output results/test.npz

Evaluation expects four canonical class directories (English or original Polish names). Probability order is derived from checkpoint names. Unknown classes and unreadable images cause errors rather than silent exclusions. Relative image IDs are preserved; these do not prove patient independence.
Do not change train/validation/test membership in response to performance. New experiments must retain their partition and report their own results.

## Verification

    python -m unittest discover -s tests
    python scripts/regenerate_probability_figures.py --inputs data --output results/figures

Figure regeneration requires internal_probs.npz and external_probs.npz in data. It exports TIFF Figures 2–4 and probability/calibration audits. Eigen-CAM reproduction is not provided without the original weights and images.

## Limits

Supplied-array metrics and checkpoint metadata were audited; independent inference on original images was not performed. Original split algorithm and patient identities were not independently verified. DISH was an exclusion criterion; severe degeneration was not. The frequency and subgroup-specific performance of severe degeneration were not established in the reported analysis. No subgroup fairness, prospective clinical utility or assisted-reader benefit was established. Weighted kappa and code distances are secondary coding-dependent descriptions. Paired comparisons require aligned IDs, labels and class order; scripts/compare_predictions.py rejects mismatches.

## Citation and releases

Current release: [v1.1.2](https://github.com/mateuszjaskowski-lgtm/axspa-yolo-cls/releases/tag/v1.1.2). See [CITATION.cff](CITATION.cff) for citation metadata.

Zenodo all-version record: https://doi.org/10.5281/zenodo.22850851. When citing an archived snapshot, check that its version matches the code used.

## Ethics and licensing

Complete ethics identifiers, scope and consent/waiver details from actual institutional decisions in the manuscript. No assumption of two separate approvals is made.
Existing MIT license is retained for repository code. Third-party frameworks/weights retain their own licenses. Repository licensing does not grant rights to patient data.
Research use only; no clinical deployment provided.

Contact: Mateusz Jaśkowski — mjaskowski@zoz.konskie.pl
