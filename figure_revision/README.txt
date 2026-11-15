FIGURE DISPLAY REVISION — 15 November 2026

Figures 2 and 4 were regenerated from saved seed-0 YOLO11l-cls probabilities. The change adds matched zoomed views; the full ROC axes, AUC values, bootstrap intervals and validation-selected threshold are retained. No model fitting or model inference was performed. See roc_revision_verification.json for input hashes and numeric checks.

After extracting Supplementary_Computational_Materials.zip, run:
python revise_roc_figures.py --source /path/to/axspa-yolo-cls-v2.1.0/analysis/yolo11l-cls/seed_0 --output /path/to/output

Figure S4 reuses the four correctly classified source images already selected for Figure 5a–d. It displays the complete prepared PNGs without CAM overlays. Source identifiers and hashes are in Figure_S4_source_ledger.json. The private study PNGs are required to regenerate this figure; they are available under the manuscript's data-access procedure and are not included as a complete dataset in this package.

python build_morphology_examples.py --dataset-root /path/to/png_root --panel-ledger /path/to/eigen_cam_panels.csv --manifest /path/to/clean_manifest.csv --output-dir /path/to/output

Use a PNG root that directly contains test/normal, test/osteofity, test/parasyndesmofity and test/syndesmofity. See the scripts' --help for arguments. Figure S4 remains in its author-approved form without lesion arrows; category labels come from the retained consensus reference. Additional localization annotations are not planned.
