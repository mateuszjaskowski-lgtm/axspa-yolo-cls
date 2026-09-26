# Model card

Task: classification of four dominant morphological patterns on lateral spinal radiographs: normal, osteophytes, parasyndesmophytes, syndesmophytes. This is not clinical axSpA diagnosis.

Primary model: ImageNet-pretrained YOLO11l-cls, Ultralytics 8.3.246, 640-pixel input. Checkpoint records batch 100, 240 epochs, seed 0; best validation top-1 at epoch 199. Historical runtime/hardware records were not verified.

Data: January 2022–December 2025; Lodz 582 training and 95 validation patients; Konskie 150 external patients. One image per patient, most painful segment selected. Original split algorithm is not independently reproduced.

Preprocessing: anonymized PNGs without text; complete PNG files as received were supplied to the model pipeline without manual cropping or additional author-performed preprocessing (including intensity adjustment or CLAHE). Framework training transforms are detailed in the manuscript. Acquisition device metadata are unavailable.

Reference: independent two-reader assessment, followed by consensus on all 174 disagreements. Dominant type by affected vertebrae; ties by individual outgrowths. Syndesmophyte class restricted to bridging lesions. Reader agreement 653/827; nominal kappa 0.702739217. Aggregate agreement and audit code included; case-level annotations are not public in this repository.

Verified supplied-array performance: accuracy 0.947368/0.900000 internal/external; macro-AUC 0.981891/0.987043; binary AUC 0.996479/0.989931. See probability audits for uncertainty and provenance.

Revision benchmark: ResNet50 140/150, EfficientNet-B0 137/150; new archived YOLO26 138/150 (nominal kappa 0.885226041). Earlier YOLO26 results are historical and superseded. New run metadata and identified predictions are in the author-held revision experiment archive; weights remain in the separate original run archive.

Limitations: selected symptomatic population; DISH was an exclusion criterion; severe degeneration was not, and its frequency and subgroup-specific performance were not established; no fairness/subgroup analysis or prospective clinical utility evaluation; internal validation used for checkpoint selection; external threshold optimization and revision benchmarking are exploratory. Paired cross-model inference is limited by missing identifiers for older arrays.

Provenance: predictions and reader metrics were reanalyzed, not independently regenerated from original radiographs. The author confirms the supplied YOLO11 checkpoint produced the evaluated results. Original radiographs and model weights are not distributed in this package. Public code and aggregate audits do not grant permission to access or reuse patient-level data. Research use only; no deployment application provided.

Additional author-confirmed limitations: separate train/validation age and sex summaries unavailable; analysis plan agreed verbally without a written plan available for review. No included image was omitted from model evaluation. No cases of DISH, prior spinal surgery/instrumented fusion, or vertebral fractures unrelated to axSpA were present in the assessed pools of 686 Lodz and 155 Konskie examinations. Only the 9 and 5 image-quality exclusions occurred in those pools; no counts are inferred for earlier screening.
