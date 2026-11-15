#!/usr/bin/env python3
"""Re-layout Figures 2 and 4 from frozen probabilities; no model inference or fitting.

The insets repeat the unchanged empirical curves on identical expanded axes.
All displayed AUCs and operating points are verified against archived metrics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve


BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE / 'sources/axspa-yolo-cls-v2.1.0-ready/axspa-yolo-cls-v2.1.0/analysis/yolo11l-cls/seed_0'
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=SOURCE)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

CLASSES = ['normal', 'osteofity', 'parasyndesmofity', 'syndesmofity']
LABELS = ['Normal', 'Osteophytes', 'Parasyndesmophytes', 'Syndesmophytes']
COLORS = ['#2368A2', '#C45C19', '#19836F', '#8F58A5']
STYLES = ['-', '--', '-.', ':']
SPLITS = ['val', 'test']
TITLES = {'val': 'Internal validation (n = 95)', 'test': 'External test (n = 150)'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': .7, 'lines.linewidth': 1.6,
    'legend.fontsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

audit = {
    'primary_model': 'YOLO11l-cls', 'seed': 0,
    'operation': 'Display-only revision of Figures 2 and 4 from frozen predictions',
    'model_inference_performed': False, 'model_fitting_performed': False,
    'probability_values_modified': False, 'curve_smoothing': False,
    'inset_limits': {'false_positive_rate': [0, .20], 'true_positive_rate': [.80, 1]},
    'source_sha256_before': {}, 'source_sha256_after': {}, 'checks': {},
    'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                    'matplotlib': matplotlib.__version__, 'scikit_learn': sklearn.__version__},
    'figures': {},
}
data = {}
for split in SPLITS:
    pp = args.source / f'{split}_predictions.npz'
    mp = args.source / f'{split}_metrics.json'
    for path in [pp, mp]:
        audit['source_sha256_before'][path.name] = sha(path)
    with np.load(pp, allow_pickle=False) as archive:
        y = archive['y_true'].copy()
        prob = archive['probabilities'].copy()
        assert archive['class_names'].tolist() == CLASSES
    metrics = json.loads(mp.read_text())
    assert len(y) == (95 if split == 'val' else 150)
    assert prob.shape == (len(y), 4)
    assert np.isfinite(prob).all()
    data[split] = y, prob, metrics
    audit['checks'][split] = {'n': len(y), 'class_order': CLASSES, 'class_auc': {}}


def axes_for_panel(fig, col, split):
    left = [.080, .575][col]
    ax = fig.add_axes([left, .390, .380, .525])
    ax.set_xlim(-.012, 1.012)
    ax.set_ylim(-.012, 1.012)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel('False-positive rate (1 - specificity)', labelpad=5)
    ax.set_ylabel('True-positive rate (sensitivity)', labelpad=5)
    ax.set_title(f'{"AB"[col]}  {TITLES[split]}', pad=11, loc='center')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#D9D9D9', linewidth=.6, zorder=0)
    ax.plot([0, 1], [0, 1], color='#929292', ls='--', lw=.9, zorder=1)
    ax.add_patch(Rectangle((0, .8), .2, .2, fill=False, ec='#737373',
                           lw=.8, ls=(0, (3, 2)), zorder=2))
    inset = ax.inset_axes([.45, .17, .51, .51])
    inset.set_facecolor('white')
    inset.set_xlim(-.002, .202)
    inset.set_ylim(.798, 1.002)
    inset.set_xticks([0, .1, .2], ['0.00', '0.10', '0.20'])
    inset.set_yticks([.8, .9, 1], ['0.80', '0.90', '1.00'])
    inset.tick_params(labelsize=7.4, length=2.5, pad=2)
    inset.set_title('Zoom of upper-left region', fontsize=7.8, pad=5)
    inset.grid(color='#E2E2E2', linewidth=.5, zorder=0)
    for spine in inset.spines.values():
        spine.set_color('#777777')
        spine.set_linewidth(.6)
    return ax, inset, left


def save(fig, stem, caption):
    for suffix, dpi in [('.png', 600), ('.pdf', 600), ('_preview.png', 180)]:
        path = args.output / (stem + suffix)
        kwargs = {'metadata': {'Title': stem, 'Creator': 'revise_roc_figures.py'}} if suffix == '.pdf' else {}
        fig.savefig(path, dpi=dpi, **kwargs)
    audit['figures'][stem] = {
        'caption': caption, 'figure_inches': list(fig.get_size_inches()),
        'png_pixels': list(Image.open(args.output / (stem + '.png')).size),
        'png_dpi': 600, 'vector_pdf': True,
        'sha256': {stem + ext: sha(args.output / (stem + ext)) for ext in ['.png', '.pdf']},
    }
    plt.close(fig)


fig = plt.figure(figsize=(7.6, 5.5))
for col, split in enumerate(SPLITS):
    y, prob, metrics = data[split]
    ax, inset, left = axes_for_panel(fig, col, split)
    handles, legend_labels, aucs = [], [], []
    for k, (label, color, style) in enumerate(zip(LABELS, COLORS, STYLES)):
        fpr, tpr, _ = roc_curve(y == k, prob[:, k], drop_intermediate=False)
        auc = roc_auc_score(y == k, prob[:, k])
        expected = metrics['metrics'][CLASSES[k] + '_ovr_auc']['estimate']
        assert abs(auc - expected) < 1e-12
        h, = ax.plot(fpr, tpr, color=color, ls=style, zorder=3)
        inset.plot(fpr, tpr, color=color, ls=style, lw=1.5, zorder=3)
        handles.append(h)
        legend_labels.append(f'{label}: AUC {auc:.3f}')
        aucs.append(auc)
        audit['checks'][split]['class_auc'][CLASSES[k]] = {
            'recomputed': float(auc), 'archived': expected,
            'absolute_difference': float(abs(auc - expected)), 'roc_points': len(fpr),
        }
    macro = metrics['metrics']['macro_ovr_auc']
    assert abs(np.mean(aucs) - macro['estimate']) < 1e-12
    audit['checks'][split]['macro_auc'] = float(np.mean(aucs))
    fig.text(left, .265,
             f'Macro AUC {macro["estimate"]:.3f} (95% CI {macro["ci_low"]:.3f}-{macro["ci_high"]:.3f})',
             ha='left', va='top', fontsize=8.7)
    fig.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(left - .008, .232),
               frameon=False, handlelength=2.8, labelspacing=.65, borderaxespad=0)
caption2 = ('Figure 2. One-versus-rest receiver operating characteristic (ROC) curves for the primary '
            'YOLO11l-cls model (seed 0) in (A) internal validation and (B) external test. Legends give '
            'class-specific areas under the curve (AUC); macro AUC is their unweighted mean. '
            'The 95% confidence intervals are the previously computed image-level percentile bootstrap '
            'intervals (2000 replicates) for the fixed fitted model. Insets display the same curves over '
            'false-positive rates 0.00-0.20 and true-positive rates 0.80-1.00; dashed rectangles mark '
            'the corresponding regions in the full-scale plots. The dashed diagonals are no-discrimination '
            'references. Curves were reconstructed directly from saved probabilities without smoothing.')
save(fig, 'Figure_2_ovr_roc', caption2)


fig = plt.figure(figsize=(7.6, 5.5))
threshold = data['val'][2]['binary_morphology']['threshold']['threshold']
assert threshold == .27126939594745636
audit['full_precision_validation_threshold'] = threshold
for col, split in enumerate(SPLITS):
    y, prob, metrics = data[split]
    ax, inset, left = axes_for_panel(fig, col, split)
    target = np.isin(y, [2, 3])
    score = prob[:, 2:].sum(axis=1)
    fpr, tpr, _ = roc_curve(target, score, drop_intermediate=False)
    auc = roc_auc_score(target, score)
    expected = metrics['metrics']['binary_auc']
    assert abs(auc - expected['estimate']) < 1e-12
    binary = metrics['binary_morphology']
    pred = score >= threshold
    counts = {'tp': int(np.sum(target & pred)), 'fn': int(np.sum(target & ~pred)),
              'fp': int(np.sum(~target & pred)), 'tn': int(np.sum(~target & ~pred))}
    assert all(counts[key] == binary[key] for key in counts)
    sens = binary['sensitivity']['estimate']
    spec = binary['specificity']['estimate']
    for target_ax in [ax, inset]:
        target_ax.plot(fpr, tpr, color=COLORS[0], lw=1.7, zorder=3)
        target_ax.scatter(1-spec, sens, marker='D', s=36, color=COLORS[1],
                          edgecolor='white', linewidth=.6, zorder=5, clip_on=False)
    fig.text(left, .265,
             f'AUC {auc:.3f} (95% CI {expected["ci_low"]:.3f}-{expected["ci_high"]:.3f})',
             ha='left', va='top', fontsize=8.7)
    fig.text(left, .213,
             f'Sensitivity: {counts["tp"]}/{counts["tp"]+counts["fn"]} ({sens:.1%})\n'
             f'Specificity: {counts["tn"]}/{counts["tn"]+counts["fp"]} ({spec:.1%})',
             ha='left', va='top', fontsize=8.7, linespacing=1.65)
    audit['checks'][split]['binary_auc'] = {'recomputed': float(auc),
                                          'archived': expected['estimate'],
                                          'absolute_difference': float(abs(auc-expected['estimate'])),
                                          'roc_points': len(fpr)}
    audit['checks'][split]['binary_operating_point_counts'] = counts
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker='D', linestyle='none', color=COLORS[1], markersize=6)],
           ['Validation-selected threshold = 0.271 (applied unchanged externally)'],
           loc='lower center', bbox_to_anchor=(.5, .025), frameon=False, fontsize=8.5)
caption4 = ('Figure 4. Exploratory binary morphology ROC curves for the primary YOLO11l-cls model '
            '(seed 0): (A) internal validation and (B) external test. The positive group combines '
            'parasyndesmophytes and syndesmophytes; the negative group combines normal morphology '
            'and osteophytes. The score is P(parasyndesmophytes) + P(syndesmophytes). Orange diamonds '
            'indicate the threshold 0.271, selected by maximizing Youden\'s index in internal validation '
            'and applied unchanged externally; the full-precision value is in the deposited '
            'validation_selected_threshold.json. Insets display the same curves over false-positive '
            'rates 0.00-0.20 and true-positive rates 0.80-1.00; dashed rectangles mark those regions '
            'in the full-scale plots. Confidence intervals for AUC are the saved 2000-replicate '
            'image-level bootstrap intervals. Operating-point uncertainty does not include uncertainty '
            'in selecting the validation threshold. This grouping concerns radiographic morphology '
            'and is not a clinical diagnosis of axial spondyloarthritis.')
save(fig, 'Figure_4_binary_morphology_roc', caption4)

for filename in audit['source_sha256_before']:
    audit['source_sha256_after'][filename] = sha(args.source / filename)
assert audit['source_sha256_before'] == audit['source_sha256_after']
audit['all_checks_passed'] = True
(args.output / 'roc_revision_verification.json').write_text(json.dumps(audit, indent=2) + '\n')
(args.output / 'roc_figure_captions.txt').write_text(caption2 + '\n\n' + caption4 + '\n')
print(json.dumps({'all_checks_passed': True, 'figures': list(audit['figures']),
                  'macro_auc': {s: audit['checks'][s]['macro_auc'] for s in SPLITS},
                  'binary_auc': {s: audit['checks'][s]['binary_auc']['recomputed'] for s in SPLITS}}))
