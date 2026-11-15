#!/usr/bin/env python3
"""Assemble Figure S4 from four existing, hash-verified prepared PNGs.

No lesion selection, re-labelling, cropping, image enhancement, CAM or synthetic
image creation is performed. The four examples are the same fixed-rule correct
examples already included in Figure 5, now displayed at larger size without CAM.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

CLASSES = [
    ('normal', 'Normal'),
    ('osteofity', 'Osteophytes'),
    ('parasyndesmofity', 'Parasyndesmophytes'),
    ('syndesmofity', 'Bridging syndesmophytes'),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--panel-ledger', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.panel_ledger.open(newline='') as f:
        ledger = list(csv.DictReader(f))
    with args.manifest.open(newline='') as f:
        manifest = {row['relative_path']: row for row in csv.DictReader(f)}

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 7.7))
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.025,
                        top=0.95, wspace=0.06, hspace=0.18)
    records = []
    for idx, ((class_name, display_name), ax) in enumerate(zip(CLASSES, axes.flat)):
        matches = [r for r in ledger if r['selection_group'] == 'correct'
                   and r['true_class'] == class_name]
        assert len(matches) == 1, (class_name, len(matches))
        row = matches[0]
        source = args.dataset_root / row['relative_path']
        entry = manifest[row['relative_path']]
        actual_sha = sha256(source)
        assert actual_sha == row['source_sha256'] == entry['sha256']
        assert row['image_id'] == entry['image_id']
        assert entry['split'] == 'test'
        assert entry['class_name'] == row['true_class'] == row['pred_class']
        image = Image.open(source)
        assert image.size == (int(entry['width']), int(entry['height']))
        # Match the documented RGB input convention: retain stored RGB channels,
        # discard alpha. No interpolation changes are made to the source bytes.
        rgb = image.convert('RGB')
        ax.set_facecolor('black')
        ax.imshow(rgb, interpolation='none', aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'{chr(65+idx)}  {display_name}', fontsize=10.5,
                     fontweight='bold', loc='center', pad=8)
        records.append({
            'panel': chr(65+idx),
            'display_label': display_name,
            'consensus_class': class_name,
            'relative_path': row['relative_path'],
            'image_id': row['image_id'],
            'source_sha256': actual_sha,
            'original_figure5_panel': row['panel'],
            'source_mode': image.mode,
            'source_width': image.width,
            'source_height': image.height,
            'verification': 'source bytes and label agree with panel ledger and clean manifest',
        })
    pdf = args.output_dir / 'Figure_S4_morphology_examples.pdf'
    png = args.output_dir / 'Figure_S4_morphology_examples.png'
    fig.savefig(pdf, dpi=300, metadata={
        'Title': 'Figure S4. Examples of the four consensus-defined morphology categories',
        'Subject': 'Real prepared radiographic PNGs; no CAM or lesion-localization annotation',
        'Author': 'Study authors',
    })
    fig.savefig(png, dpi=300)
    plt.close(fig)

    caption = (
        'Figure S4. Examples of the four consensus-defined morphology categories: '
        '(A) normal morphology, (B) osteophytes, (C) parasyndesmophytes and '
        '(D) bridging syndesmophytes. The panels show the full field of the '
        'prepared PNGs at larger display size, with aspect ratios preserved and '
        'without intensity enhancement or feature-map overlays. They are the same '
        'four correctly classified, fixed-rule examples shown in Figure 5a-d, '
        'selected by the first lexicographic image path within each correctly '
        'classified reference class. Labels describe the dominant morphology '
        'assigned by radiologist consensus; the figure does not provide lesion-level '
        'annotations. Source image identifiers and hashes are recorded in the '
        'accompanying Figure S4 source ledger.'
    )
    (args.output_dir/'Figure_S4_caption.txt').write_text(caption+'\n', encoding='utf-8')
    report = {
        'figure': 'S4', 'source': 'prepared study PNGs',
        'selection': 'reuse the four fixed-rule correct examples already shown in Figure 5a-d',
        'display': {'full_prepared_field': True, 'aspect_preserved': True,
                    'intensity_enhancement': False, 'cropping': False,
                    'alpha_policy': 'discard alpha and retain stored RGB, matching model input convention',
                    'cam_overlay': False, 'lesion_arrows': False,
                    'figure_size_inches': [6.2, 7.7], 'png_dpi': 300},
        'reference_manifest_sha256': sha256(args.manifest),
        'reference_panel_ledger_sha256': sha256(args.panel_ledger),
        'panels': records,
        'outputs': {p.name: sha256(p) for p in [pdf, png]},
        'all_source_checks_passed': True,
    }
    (args.output_dir/'Figure_S4_source_ledger.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'files': [str(pdf),str(png)], 'source_checks': 'PASS'},indent=2))


if __name__ == '__main__':
    main()
