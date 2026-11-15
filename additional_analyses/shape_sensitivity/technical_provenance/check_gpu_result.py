#!/usr/bin/env python3
"""Independent artifact/metric audit; never imports or runs the study inference code."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import zipfile

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score

CLASSES = ['normal', 'osteofity', 'parasyndesmofity', 'syndesmofity']
CONDITIONS = ['original', 'fixed_gray_128', 'per_image_mean_gray']
FREEZE_SHA = '2516ecb0a018afce3460e0eb0c50138aa24bb829167a354a02056d8f0590a514'
CHECKPOINT_SHA = '2ef70f6c73c43970cbaebed2884e71b1cd374b1545d5239abe6f2243e75c5ac6'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def near(a, b, message, atol=1e-12):
    require(np.allclose(a, b, rtol=0, atol=atol), message)


class Files:
    def __init__(self, path):
        self.path = Path(path)
        self.z = None if self.path.is_dir() else zipfile.ZipFile(self.path)
        if self.z:
            names = self.z.namelist()
            require(len(names) == len(set(names)), 'Duplicate ZIP member names')
            require(self.z.testzip() is None, 'ZIP CRC failure')

    def read(self, name):
        return self.z.read(name) if self.z else (self.path / name).read_bytes()

    def exists(self, name):
        return name in self.z.namelist() if self.z else (self.path / name).is_file()

    def json(self, name):
        return json.loads(self.read(name))


def npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def audit(args):
    result, package = Files(args.results), Files(args.package)
    report = {'status': 'INCOMPLETE', 'scientific_interpretation_allowed': False}
    if result.exists('failure.json') or not result.exists('summary.json'):
        report['reason'] = 'Run did not complete both prespecified conditions.'
        if result.exists('failure.json'):
            report['failure'] = result.json('failure.json')
        if result.exists('baseline_replay_gate.json'):
            report['reported_replay_gate'] = result.json('baseline_replay_gate.json')
        return report
    summary = result.json('summary.json')
    require(summary['status'] == 'COMPLETE', 'Summary status is not COMPLETE')
    require(result.read('analysis_freeze.json') == package.read('analysis_freeze.json'), 'Freeze bytes changed')
    require(sha(result.read('analysis_freeze.json')) == FREEZE_SHA, 'Wrong freeze identity')
    require(result.read('analysis_plan.json') == package.read('analysis_plan.json'), 'Plan bytes changed')
    freeze = package.json('analysis_freeze.json')
    require(sha(package.read('run_shape_sensitivity.py')) == freeze['script_sha256'], 'Frozen script mismatch')
    require(sha(package.read('analysis_plan.json')) == freeze['plan_sha256'], 'Frozen plan mismatch')
    require(sha(package.read('PLAN_ANALIZY.txt')) == freeze['readable_plan_sha256'], 'Readable plan mismatch')
    require(sha(package.read('checkpoint/selected.pt')) == CHECKPOINT_SHA, 'Checkpoint mismatch')
    for name, expected in freeze['source_hashes'].items():
        require(sha(package.read('reference_project/' + name)) == expected, 'Source mismatch: ' + name)
    hashes = result.json('output_hashes.json')
    required_hashed = {'summary.json', 'baseline_replay_gate.json', 'runtime.json', 'input_audit.json',
                       'loaded_transform_audit.json', 'transformation_audit.json', 'predictions.csv',
                       'analysis_freeze.json', 'analysis_plan.json'} | {c + '_predictions.npz' for c in CONDITIONS}
    require(required_hashed <= set(hashes), 'Required outputs absent from output hash manifest')
    for name, expected in hashes.items():
        require(sha(result.read(name)) == expected, 'Output hash mismatch: ' + name)
    runtime = result.json('runtime.json')
    require(runtime['cuda_available'] and runtime['torch_cuda'] == '12.8', 'Not CUDA12.8 GPU execution')
    for name, version in {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128', 'ultralytics': '8.4.163', 'Pillow': '11.3.0'}.items():
        require(runtime['packages'][name] == version, 'Unexpected runtime version: ' + name)
    receipt = result.json('execution_receipt.json')
    require(receipt['return_code'] == 0 and not receipt['analysis_tolerance_changed'], 'Execution receipt reports a failure or altered tolerance')
    require(receipt['freeze_sha256'] == FREEZE_SHA, 'Receipt freeze mismatch')
    require(receipt['frozen_script_sha256'] == freeze['script_sha256'], 'Receipt script mismatch')
    require(receipt['input_package_sha256'] == sha(Path(args.package).read_bytes()), 'Receipt identifies a different uploaded package')
    all_rows = rows(package.read('reference_project/data/clean_manifest.csv'))
    manifest = [r for r in all_rows if r['split'] == 'test']
    ids = [r['image_id'] for r in manifest]
    y = np.asarray([int(r['label']) for r in manifest])
    require(len(ids) == len(set(ids)) == 150, 'Wrong cohort identity/count')
    require(np.array_equal(np.bincount(y, minlength=4), [55, 54, 20, 21]), 'Wrong class counts')
    require(set(summary['conditions']) == set(CONDITIONS), 'Conditions added or omitted')
    input_audit = result.json('input_audit.json')
    require(len(input_audit) == 150 and len({r['image_id'] for r in input_audit}) == 150, 'Input audit incomplete/duplicated')
    expected_inputs = {r['image_id']: r for r in manifest}
    for r in input_audit:
        expected = expected_inputs[r['image_id']]
        require(r['relative_path'] == expected['relative_path'], 'Input audit path mismatch')
        require(r['actual_file_sha256'] == r['expected_file_sha256'] == expected['sha256'], 'Input audit file hash mismatch')
        require(r['independent_rgb_sha256'] == expected['independent_rgb_sha256'] and r['file_bytes_match'] and r['decoded_rgb_and_dimensions_match'], 'Input audit RGB/identity check failed')
    require(result.json('loaded_transform_audit.json')['whole_png_aspect_preserving_pad_640_verified'], 'Loaded checkpoint transform was not verified')
    original_means = {}
    for r in manifest:
        data = package.read('dataset/' + r['relative_path'])
        require(sha(data) == r['sha256'], 'Original PNG bytes mismatch')
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert('RGB')
            require(image.size == (int(r['width']), int(r['height'])), 'PNG geometry mismatch')
            pixel_hash = sha(f'RGB:{image.width}x{image.height}:'.encode() + image.tobytes())
            require(pixel_hash == r['independent_rgb_sha256'], 'PNG RGB hash mismatch')
            original_means[r['image_id']] = float(np.asarray(image, dtype=np.float64).mean())
    reference = npz(package.read('reference_project/analysis/yolo11l-cls/seed_0/test_predictions.npz'))
    lookup = {str(v): i for i, v in enumerate(reference['image_id'])}
    reference_p = reference['probabilities'][[lookup[v] for v in ids]]
    require(list(reference['class_names']) == CLASSES, 'Reference class order mismatch')
    predictions, recomputed = {}, {}
    z = NormalDist().inv_cdf(.975)
    for condition in CONDITIONS:
        a = npz(result.read(condition + '_predictions.npz'))
        require(list(a['image_id']) == ids, condition + ': image ID order mismatch')
        require(list(a['sha256']) == [r['sha256'] for r in manifest], condition + ': source hash mapping mismatch')
        require(list(a['class_names']) == CLASSES and np.array_equal(a['y_true'], y), condition + ': class/label mismatch')
        p = np.asarray(a['probabilities'], dtype=float)
        require(p.shape == (150, 4) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, condition + ': invalid probabilities')
        near(p.sum(1), np.ones(150), condition + ': probability sums', atol=1e-5)
        pred = p.argmax(1); correct = int((pred == y).sum()); acc = accuracy_score(y, pred)
        center = (acc + z*z/300)/(1+z*z/150)
        half = z*math.sqrt(acc*(1-acc)/150+z*z/(4*150**2))/(1+z*z/150)
        m = summary['conditions'][condition]
        require(m['accuracy']['correct'] == correct and m['accuracy']['total'] == 150, condition + ': count mismatch')
        near([m['accuracy']['estimate'], m['accuracy']['ci_low'], m['accuracy']['ci_high']], [acc, center-half, center+half], condition + ': accuracy/Wilson mismatch')
        values = {'balanced_accuracy': balanced_accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, labels=range(4), average='macro', zero_division=0), 'unweighted_cohen_kappa': cohen_kappa_score(y, pred, labels=range(4)), 'mean_max_probability': p.max(1).mean()}
        for name, value in values.items(): near(m[name], value, condition + ': ' + name)
        near(m['confusion_matrix_true_rows_predicted_columns'], confusion_matrix(y, pred, labels=range(4)), condition + ': confusion matrix')
        near(m['predicted_class_counts'], np.bincount(pred, minlength=4), condition + ': prediction counts')
        near(m['true_class_counts'], np.bincount(y, minlength=4), condition + ': true counts')
        near(m['class_sensitivity'], recall_score(y, pred, labels=range(4), average=None, zero_division=0), condition + ': recalls')
        predictions[condition] = p
        recomputed[condition] = {'correct': correct, 'accuracy': float(acc), 'wilson_ci': [center-half, center+half], **{k: float(v) for k, v in values.items()}}
    delta = np.abs(predictions['original'] - reference_p)
    gate = result.json('baseline_replay_gate.json')
    require(int((predictions['original'].argmax(1) == y).sum()) == 138, 'Original replay accuracy differs')
    require(np.array_equal(predictions['original'].argmax(1), reference_p.argmax(1)), 'Original replay decisions differ')
    require(delta.max() <= 1e-4 and gate['status'] == 'PASS', 'Original replay probability gate failed')
    near(gate['maximum_absolute_probability_difference'], delta.max(), 'Reported replay maximum mismatch')
    near(gate['mean_absolute_probability_difference'], delta.mean(), 'Reported replay mean mismatch')
    require(gate['prespecified_absolute_tolerance'] == 1e-4 and gate['argmax_matches'] == 150, 'Replay tolerance/count changed')
    bootstrap_indices = np.random.default_rng(0).integers(0, 150, size=(10000, 150))
    base_correct = predictions['original'].argmax(1) == y
    for condition in CONDITIONS[1:]:
        variant_correct = predictions[condition].argmax(1) == y
        samples = base_correct[bootstrap_indices].mean(1) - variant_correct[bootstrap_indices].mean(1)
        m = summary['paired_comparisons'][condition]
        near([m['estimate_original_minus_variant'], m['ci_low'], m['ci_high']], [base_correct.mean()-variant_correct.mean(), *np.quantile(samples, [.025, .975])], condition + ': paired difference/CI')
        require(m['replicates'] == 10000 and m['seed'] == 0, 'Bootstrap configuration changed')
        agreement = int((predictions['original'].argmax(1) == predictions[condition].argmax(1)).sum())
        require(m['prediction_agreement_count_with_original'] == agreement, 'Prediction agreement count mismatch')
        near(m['prediction_agreement_with_original'], agreement/150, 'Prediction agreement percentage mismatch')
    flat = rows(result.read('predictions.csv'))
    require(len(flat) == 450 and len({(r['condition'], r['image_id']) for r in flat}) == 450, 'CSV incomplete or duplicated')
    for r in flat:
        i = ids.index(r['image_id']); p = predictions[r['condition']][i]
        require(int(r['y_true']) == int(y[i]) and int(r['prediction']) == int(p.argmax()), 'CSV label/prediction mismatch')
        near([float(r['p_'+c]) for c in CLASSES], p, 'CSV/NPZ probabilities differ', atol=1e-15)
    transforms = result.json('transformation_audit.json')
    require(len(transforms) == 300 and len({(r['condition'], r['image_id']) for r in transforms}) == 300, 'Transformation audit incomplete or duplicated')
    by_id = {r['image_id']: r for r in manifest}
    for r in transforms:
        source = by_id[r['image_id']]; w, h = int(source['width']), int(source['height'])
        require(r['condition'] in CONDITIONS[1:] and (r['width'], r['height']) == (w, h), 'Transformation condition/dimensions changed')
        mean = original_means[r['image_id']]; gray = 128 if r['condition'] == 'fixed_gray_128' else math.floor(mean+.5)
        require(r['uniform_gray'] == gray, 'Incorrect gray value')
        near(r['original_mean_rgb_unrounded'], mean, 'Incorrect original mean')
        scale = 640/max(w,h)
        rw, rh = max(1, min(640, round(w*scale))), max(1, min(640, round(h*scale)))
        left, top = (640-rw)//2, (640-rh)//2
        require([r[k] for k in ['resized_width','resized_height','padding_left','padding_top','padding_right','padding_bottom']] == [rw,rh,left,top,640-rw-left,640-rh-top], 'Footprint/padding changed')
        pixels = bytes([gray])*(w*h*3)
        require(r['variant_independent_rgb_sha256'] == sha(f'RGB:{w}x{h}:'.encode()+pixels), 'Uniform RGB identity differs')
        require(r['unchanged_footprint_and_zero_padding_verified'], 'Footprint check was not passed')
    require(summary['checkpoint_sha256'] == CHECKPOINT_SHA and summary['freeze_sha256'] == FREEZE_SHA and summary['plan_sha256'] == freeze['plan_sha256'], 'Summary provenance mismatch')
    train_labels = [int(r['label']) for r in all_rows if r['split'] == 'train']
    train_counts = np.bincount(train_labels, minlength=4)
    require(len(train_labels) == 579 and int(train_counts.argmax()) == 0, 'Training majority baseline changed')
    comparators = summary['descriptive_comparators']
    near(comparators['training_class_counts'], train_counts, 'Training class counts mismatch')
    near(comparators['training_majority_class_normal_accuracy'], 55/150, 'Majority baseline accuracy mismatch')
    near(comparators['training_majority_class_normal_balanced_accuracy'], .25, 'Majority baseline balanced accuracy mismatch')
    near(comparators['previous_geometry_brightness_logistic_accuracy'], 67/150, 'Previous geometry classifier reference mismatch')
    report.update(status='PASS', scientific_interpretation_allowed=True, n=150, original_replay_max_abs=float(delta.max()), conditions=recomputed, checks=['CRC and file hashes', 'frozen code/plan/checkpoint identity', 'original PNG identities', 'GPU CUDA12.8 runtime', '150 original decisions and probability gate', '450 predictions CSV/NPZ', 'sklearn point metrics and Wilson intervals', 'paired bootstrap independently recomputed', '300 geometry/gray transformations'], caveat='Artifact consistency does not prove absence of shortcut learning; apply the frozen OOD interpretation limits.')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True, help='Downloaded result ZIP or extracted result directory')
    p.add_argument('--package', required=True, help='Exact private ZIP that was actually uploaded to Colab')
    p.add_argument('--output', required=True, help='New audit JSON; existing files are never overwritten')
    args = p.parse_args(); out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    try: report = audit(args)
    except Exception as e: report = {'status': 'FAIL', 'scientific_interpretation_allowed': False, 'error': type(e).__name__ + ': ' + str(e)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    sys.exit(0 if report['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
