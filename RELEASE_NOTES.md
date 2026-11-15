# Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model

## Materials version 2.1.0 — 8 November 2026

This release aligns the public reproducibility package with the current manuscript prepared for *Diagnostics*. It contains the fixed 824-image experiment and its completed supplementary analyses. It does not report a new training experiment or a new independent external validation.

- **Cohort:** 674 development cases from Łódź (579 training, 95 validation) and 150 external cases from Końskie. The cohort is defined by chronic back pain and referral by a rheumatologist. The author-confirmed eligibility flow is 716−33−9=674 and 163−8−5=150; severe degenerative changes were not excluded. Figure S1 and its generating code use this accounting.
- **Allocation:** manual within-class numerical filename ranges, selected without regard to image content and fixed before first training. Acquisition chronology was not established from those filenames.
- **Supplementary analyses:** all code, saved non-image outputs and execution records for geometry/brightness (S7), initial readers versus their own consensus (S8), and gray-content ablation (S9). Their post hoc exploratory status and limits are explicit.
- **Interpretation:** the same readers labeled both cohorts; seed 0 happened to be the best external YOLO11 run, while the five-seed mean was 88.4%; gray controls do not establish reliance specifically on bone features. The replay requirement is a maximum absolute probability difference no greater than 0.0001.
- **Technical descriptions:** shared stored YOLO backbone/head definitions and different pretrained checkpoint hashes are distinguished from claims about tensor differences. Exact-image uniqueness is described neutrally. The displayed binary threshold is 0.271, with full precision retained in its deposited JSON.
- **Current reproduction:** active preparation accepts the registered 824-image archives and verifies accepted memberships and identities. Numerical model training/evaluation methods are unchanged. Original executed source and frozen artifacts remain identifiable as execution evidence, separate from current study descriptions.
- **Publication material:** the complete current supplementary methods and results PDF is supplied. The main manuscript, administrative submission files, source radiographs, checkpoint binaries and private CAM arrays are not part of this software deposit.

The primary result remains 138/150 external classifications correct (92.0%; exact 95% CI 86.4–95.8), and the YOLO11 five-seed mean remains 88.4%. Model predictions, fitted checkpoint hashes, consensus labels and fixed split memberships are unchanged. The release adds the already completed gray-control predictions as a distinct sensitivity analysis.

See `CITATION.cff` for citation metadata and the [Zenodo all-versions DOI](https://doi.org/10.5281/zenodo.22850851) for the deposit series. The version-specific record identifies the archived package and should be used for exact-version citation.

The [15 November review revision](REVISION_ADDENDUM_2026-11-15.md) documents the methods and ethics clarifications, current correspondence details, revised ROC figures and the Figure S4 reconstruction procedure. The public PDF contains the supplementary methods, tables and Figures S1–S3. Figure S4 is reproduced from the study images obtained through the manuscript’s data-access procedure.
