# axSpA Radiographic Changes — YOLO11l-cls

Deep learning model for detection of radiographic changes on lateral spinal radiographs in patients evaluated for axial spondyloarthritis (axSpA). The model classifies radiographs into four morphological categories of structural change.

This repository accompanies the manuscript:

> **Jaśkowski M, Górski P, Guzera Z, Podgórski M.** Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model. *Diagnostics* (submitted, 2026).

---

## Overview

The model classifies lateral spinal radiographs into four ordinal categories:

| Category | Description |
|----------|-------------|
 Normal | Absence of structural lesions |
 Osteophytes | Horizontally-oriented bony outgrowths (degenerative) |
 Parasyndesmophytes | Bridging outgrowths of intermediate morphology |
 Syndesmophytes | Vertically-oriented bony bridges across the disc space (axSpA-related) |

## Performance

Validated on an independent external cohort (n=150) from a structurally distinct healthcare center:

| Metric | Value | 95% CI |
|--------|-------|--------|
| Overall accuracy | 0.900 | 0.847–0.947 |
| Quadratic-weighted Cohen's κ | 0.907 | 0.825–0.963 |
| Macro AUC | 0.989 | — |
| Within-1-grade accuracy | 0.980 | — |
| **Binary task AUC** (axSpA-related vs non-axSpA) | **0.990** | **0.975–0.999** |

## Model architecture

- **Base model:** YOLO11l-cls (Ultralytics)
- **Parameters:** ~13M
- **Input:** 640×640 pixel grayscale/RGB images
- **Pretraining:** ImageNet-1K
- **Training framework:** Ultralytics 8.3.246 (PyTorch 2.10.0, CUDA 12.8)
- **Hardware:** Single NVIDIA Tesla T4 GPU (16 GB VRAM)
- **Training time:** ~16 minutes for 240 epochs
- **Inference time:** ~28 ms per image (34.7 images/second on Tesla T4)

## Repository structure

```
.
├── README.md                    # This file
├── LICENSE                       # MIT License
├── CITATION.cff                  # Citation metadata
├── requirements.txt              # Python dependencies
├── notebooks/
│   ├── analysis.ipynb            # Main analysis notebook (inference, metrics, ROC, calibration, Eigen-CAM)
│   └── inter_rater_analysis.ipynb  # Inter-rater agreement analysis
├── scripts/
│   ├── inference.py              # Minimal single-image inference script
│   └── export_metrics.py         # Metrics export script for validation folder
├── docs/
│   ├── model_card.md             # Model card (intended use, limitations, ethics)
│   └── training_config.yaml      # Training hyperparameters
└── example_outputs/
    └── example_predictions.md    # Example predictions with confidence scores
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Single-image inference

```bash
python scripts/inference.py --image path/to/radiograph.jpg --model best.pt
```

Output:
```
Predicted grade: 2 (Parasyndesmophytes)
Confidence: 0.87
Class probabilities:
  (Normal):            0.03
  (Osteophytes):       0.08
  (Parasyndesmophytes): 0.87
  (Syndesmophytes):    0.02
```

### 3. Batch evaluation

Use the Colab notebook `notebooks/analysis.ipynb` for full evaluation including confusion matrices, ROC curves, calibration analysis, and Eigen-CAM visualizations.

## Model weights

Model weights (best.pt) are available upon reasonable request from the corresponding author due to institutional and data-protection constraints on the training data.

For research collaboration or clinical evaluation, please contact:

**Mateusz Jaśkowski, MD**
Rheumatology Department, Saint Lucas Hospital
ul. Gimnazjalna 41B, 26-200 Konskie, Poland
Email: mjaskowski@zoz.konskie.pl
ORCID: [0009-0003-7303-9600](https://orcid.org/0009-0003-7303-9600)

## Ethics and data availability

- The study was approved by the institutional Ethics Committees of both participating centers (Medical University of Lodz and Saint Lucas Hospital, Konskie)
- Individual informed consent was waived due to the retrospective design and analysis of irreversibly anonymized imaging data
- Individual-patient imaging data are not publicly available due to institutional and data-protection constraints
- Aggregate results and metrics are provided in the manuscript and this repository

## Intended use and limitations

**Intended use:** Research and development of AI-assisted radiographic interpretation tools for axSpA imaging.

**Not intended for:** Standalone clinical decision-making without expert radiologist supervision.

See [`docs/model_card.md`](docs/model_card.md) for full model card including limitations, ethical considerations, and out-of-scope uses.

## Citation

If you use this model or code in your research, please cite:

```bibtex
@article{jaskowski2026axspa,
  title={Radiographic Differentiation of Axial Spondyloarthritis and Degenerative Spinal Changes in Patients with Chronic Back Pain: Development and External Validation of a YOLO-Based Deep Learning Model},
  author={Jaśkowski, Mateusz and Górski, Paweł and Guzera, Zbigniew and Podgórski, Michał},
  journal={Diagnostics},
  year={2026},
  publisher={MDPI}
}
```

## License

This repository is licensed under the [MIT License](LICENSE).

The pretrained YOLO11l-cls base model is licensed under [AGPL-3.0 by Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## Acknowledgments

- Ultralytics team for the YOLO11 framework
- All patients whose de-identified imaging contributed to this research

## Contact

For questions, issues, or collaboration inquiries, please open an [issue](../../issues) or contact the corresponding author.
