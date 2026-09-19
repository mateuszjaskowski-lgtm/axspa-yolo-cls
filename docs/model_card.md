# Model Card — YOLO11l-cls for axSpA Radiographic Changes

## Model details

- **Architecture:** YOLO11l-cls (Ultralytics)
- **Model size:** ~13 million parameters
- **Framework:** Ultralytics 8.3.246 (PyTorch 2.10.0, CUDA 12.8)
- **Pretraining:** ImageNet-1K
- **Input:** 640×640 pixel images (grayscale converted to 3-channel)
- **Output:** Softmax probabilities over 4 ordinal classes
- **Training seed:** 0 (deterministic mode enabled)
- **Training time:** ~16 minutes on NVIDIA Tesla T4 (16 GB VRAM)
- **Inference speed:** ~28 ms/image, 34.7 images/second (Tesla T4)
- **Release date:** 2026
- **Version:** 1.0.0

## Intended use

### Primary intended use
Research and development of AI-assisted radiographic interpretation tools for axial spondyloarthritis (axSpA). The model is designed to classify lateral spinal radiographs into four ordinal categories corresponding to structural changes ranging from normal findings to inflammatory syndesmophytes.

### Primary intended users
- Musculoskeletal radiologists and rheumatologists conducting AI research
- Developers building clinical decision support prototypes
- Medical imaging researchers evaluating deep learning approaches for axSpA

### Out-of-scope uses
- **Standalone clinical decision-making without expert radiologist supervision**
- Diagnosis of conditions other than axSpA-related versus degenerative spinal changes
- Pediatric radiographs (training population was adult, age ≥18)
- Anteroposterior (AP) radiographs — model was trained exclusively on lateral views
- CT, MRI, or ultrasound imaging
- Radiographs of anatomical regions other than the spine

## Training data

- **Source:** Medical University of Lodz, Poland (development cohort)
- **Size:** 677 patients, 677 lateral spinal radiographs
- **Enrollment period:** January 2022 – February 2026
- **Spinal segments:** Lumbosacral (n=228), thoracic (n=229), cervical (n=220)
- **Class distribution:**
  - Grade 0 (Normal): 250 (36.9%)
  - Grade 1 (Osteophytes): 245 (36.2%)
  - Grade 2 (Parasyndesmophytes): 93 (13.7%)
  - Grade 3 (Syndesmophytes): 89 (13.1%)
- **Train/validation split:** 582/95 (patient-level random split)
- **Data augmentation:** Ultralytics default (fliplr=0.5, mosaic=1, HSV perturbations, RandAugment, random erasing 0.4); vertical flip disabled (flipud=0)

## Evaluation data

External test cohort held out throughout development:
- **Source:** Saint Lucas Hospital, Konskie, Poland (structurally distinct district rheumatology hospital)
- **Size:** 150 patients
- **Enrollment period:** January 2022 – February 2026
- **Class distribution:** Grade 0: 55, Grade 1: 54, Grade 2: 20, Grade 3: 21

## Performance metrics

### Internal validation (n=95)
- Accuracy: 0.947 (95% CI 0.895–0.989)
- Quadratic-weighted κ: 0.974 (95% CI 0.944–0.995)
- Macro AUC: 0.985
- Within-1-grade accuracy: 1.000

### External test (n=150)
- Accuracy: 0.900 (95% CI 0.847–0.947)
- Quadratic-weighted κ: 0.907 (95% CI 0.825–0.963)
- Macro AUC: 0.989
- Within-1-grade accuracy: 0.980
- Brier score: 0.038

### Binary task (axSpA-related vs non-axSpA-related)
- AUC: 0.990 (95% CI 0.975–0.999)
- At Youden's optimum (threshold 0.099):
  - Sensitivity: 0.951
  - Specificity: 0.954
  - PPV: 0.886
  - NPV: 0.981

## Ethical considerations

- The model was trained on data collected from adult patients with clinical suspicion of axSpA
- The training population was exclusively Polish; generalization to other populations may require further validation
- The model output should never replace expert radiologist interpretation
- Model confidence at the individual-case level should not be used in isolation to defer difficult cases to expert review — individual misclassifications occasionally occurred with high predicted probability
- No demographic bias analysis was performed; further work is needed to assess performance across sex, age, and disease severity subgroups

## Known limitations

1. **No radiologist head-to-head comparison** — the model has not been formally benchmarked against individual radiologists of varying experience levels
2. **No architecture benchmarking** — only YOLO11l-cls was evaluated; comparative performance against ResNet, EfficientNet, or vision transformers is unknown
3. **Region-agnostic pooling** — the model was trained on lumbosacral, thoracic, and cervical radiographs pooled together; per-segment stratified performance was not evaluated
4. **Retrospective single-country dataset** — multinational validation is needed to establish broader applicability
5. **Reference standard limitations** — two-reader consensus rather than independent multi-reader scoring (pre-consensus quadratic-weighted κ=0.828)
6. **Overconfidence on individual errors** — despite good overall calibration (Brier=0.038), individual misclassifications occasionally occurred with high predicted probability
7. **No prospective clinical validation** — real-world impact on diagnostic workflow and patient outcomes has not been evaluated

## Compute environment

- **Training hardware:** NVIDIA Tesla T4 GPU (16 GB VRAM)
- **Software:** Python 3.12, Ultralytics 8.3.246, PyTorch 2.10.0, CUDA 12.8
- **Training environment:** Ultralytics HUB with Automatic Mixed Precision (AMP) enabled
- **Reproducibility:** Random seed fixed at 0; deterministic training mode enabled

## References

- Manuscript: Jaśkowski M, Górski P, Guzera Z, Podgórski M. Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model. Diagnostics (submitted, 2026).
- YOLO11 architecture: Khanam R, Hussain M. YOLOv11: an overview of the key architectural enhancements. arXiv:2410.17725 (2024).

## Contact

**Corresponding author:** Mateusz Jaśkowski, MD
Rheumatology Department, Saint Lucas Hospital
ul. Gimnazjalna 41B, 26-200 Konskie, Poland
Email: mjaskowski@zoz.konskie.pl
ORCID: 0009-0003-7303-9600
