#!/usr/bin/env python3
"""Post-hoc exploratory image-geometry/brightness analysis; no CNN execution.

Run from any working directory. Optional positional arguments are PNG ZIP,
canonical manifest CSV, output directory. The frozen plan must exist in output.
Requires numpy, pandas, scipy, scikit-learn and Pillow; see frozen plan versions.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import io
import json
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kruskal, norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ARCHIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'recovered_extra/baza danych(1).zip'
MANIFEST = Path(sys.argv[2]) if len(sys.argv) > 2 else WORKSPACE / 'release_2026-10-25/repo/data/clean_manifest.csv'
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE
FEATURES = ['width_px', 'height_px', 'aspect_ratio_width_over_height', 'mean_rgb_brightness_0_1']
LABELS = [0, 1, 2, 3]
NAMES = ['normal', 'osteophytes', 'parasyndesmophytes', 'syndesmophytes']


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def metrics(y_true, y_pred):
    return {'n': int(len(y_true)), 'correct': int(np.sum(np.asarray(y_true) == np.asarray(y_pred))),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)),
            'confusion_matrix_true_rows_predicted_columns': confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}


def wilson(k, n):
    z = float(norm.ppf(0.975))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / 'analysis_plan_frozen_before_execution.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = file_sha256(plan_path)
    assert file_sha256(ARCHIVE) == plan['inputs']['archive_sha256']
    assert file_sha256(MANIFEST) == plan['inputs']['manifest_sha256']
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest) == 824
    assert manifest['relative_path'].is_unique and manifest['image_id'].is_unique
    assert manifest['split'].value_counts().to_dict() == {'train': 579, 'test': 150, 'val': 95}
    records = []
    with zipfile.ZipFile(ARCHIVE) as z:
        mapping = {}
        for n in z.namelist():
            if n.lower().endswith('.png'):
                parts = Path(n).parts
                for idx, part in enumerate(parts):
                    if part in ('train', 'val', 'test'):
                        key = '/'.join(parts[idx:])
                        assert key not in mapping
                        mapping[key] = n
                        break
        assert set(mapping) == set(manifest['relative_path'])
        for row in manifest.to_dict('records'):
            raw = z.read(mapping[row['relative_path']])
            assert hashlib.sha256(raw).hexdigest() == row['sha256']
            with Image.open(io.BytesIO(raw)) as im:
                rgb = im.convert('RGB')
                width, height = rgb.size
                assert width == row['width'] and height == row['height']
                brightness = float(np.asarray(rgb, dtype=np.float64).mean() / 255.0)
            records.append({'image_id': row['image_id'], 'relative_path': row['relative_path'],
                            'split': row['split'], 'label': int(row['label']),
                            'class_name': NAMES[int(row['label'])], 'sha256': row['sha256'],
                            'width_px': width, 'height_px': height,
                            'aspect_ratio_width_over_height': width / height,
                            'mean_rgb_brightness_0_1': brightness})
    features = pd.DataFrame(records)
    features.to_csv(OUT / 'image_geometry_brightness_features.csv', index=False)
    assert np.isfinite(features[FEATURES].to_numpy()).all()
    train = features[features['split'] == 'train']
    pipeline = Pipeline([('scale', StandardScaler()),
                         ('logistic', LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000,
                                                        random_state=0, class_weight=None))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(train[FEATURES].to_numpy(), train['label'].to_numpy())
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError('Predefined model did not converge: ' + '; '.join(convergence))
    train_counts = np.bincount(train['label'], minlength=4)
    majority_label = int(np.argmax(train_counts))
    split_results = {}
    prediction_records = []
    summary_rows = []
    kw_rows = []
    for split in ['train', 'val', 'test']:
        part = features[features['split'] == split]
        probs = pipeline.predict_proba(part[FEATURES].to_numpy())
        pred = np.argmax(probs, axis=1)
        baseline = np.full(len(part), majority_label)
        actual = part['label'].to_numpy()
        result = {'class_counts': np.bincount(actual, minlength=4).tolist(),
                  'logistic': metrics(actual, pred), 'train_majority_baseline': metrics(actual, baseline)}
        if split == 'test':
            for k in ['logistic', 'train_majority_baseline']:
                result[k]['accuracy_ci95_wilson'] = wilson(result[k]['correct'], len(part))
        split_results[split] = result
        for row, prediction, probabilities in zip(part.to_dict('records'), pred, probs):
            record = {k: row[k] for k in ['image_id', 'relative_path', 'split', 'label', 'class_name']}
            record['predicted_label'] = int(prediction)
            record['predicted_class'] = NAMES[prediction]
            record['majority_baseline_label'] = majority_label
            for label, prob in enumerate(probabilities):
                record['prob_' + NAMES[label]] = float(prob)
            prediction_records.append(record)
        for label in LABELS:
            part_class = part[part['label'] == label]
            for feature in FEATURES:
                q = part_class[feature].quantile([.25, .5, .75], interpolation='linear').to_numpy()
                summary_rows.append({'split': split, 'label': label, 'class_name': NAMES[label],
                                     'feature': feature, 'n': len(part_class),
                                     'q1': q[0], 'median': q[1], 'q3': q[2], 'iqr': q[2] - q[0]})
        tests = []
        for feature in FEATURES:
            statistic, pvalue = kruskal(*[part[part['label'] == label][feature].to_numpy() for label in LABELS])
            tests.append({'split': split, 'feature': feature, 'n': len(part),
                          'H_statistic': float(statistic), 'df': 3, 'p_raw': float(pvalue)})
        for test, padj in zip(tests, holm([test['p_raw'] for test in tests])):
            test['p_holm_within_split_four_features'] = padj
            kw_rows.append(test)
    pd.DataFrame(prediction_records).to_csv(OUT / 'logistic_predictions.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / 'class_feature_descriptives.csv', index=False)
    pd.DataFrame(kw_rows).to_csv(OUT / 'kruskal_wallis_holm.csv', index=False)
    result = {'analysis_status': plan['analysis_status'], 'execution_completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'plan_sha256': plan_hash, 'code_sha256': file_sha256(Path(__file__)),
              'input_archive_sha256': file_sha256(ARCHIVE), 'input_manifest_sha256': file_sha256(MANIFEST),
              'all_824_png_byte_hashes_and_dimensions_verified': True, 'features': FEATURES,
              'classes': NAMES, 'versions': plan['versions'], 'train_class_counts': train_counts.tolist(),
              'baseline_fixed_class': {'label': majority_label, 'name': NAMES[majority_label]},
              'uniform_four_class_chance_context': .25, 'split_results': split_results,
              'kruskal_wallis_tests': kw_rows, 'model': {'type': 'multinomial_logistic_regression',
              'C': 1.0, 'solver': 'lbfgs', 'max_iter': 5000, 'n_iter': pipeline['logistic'].n_iter_.tolist(),
              'standardization_fitted_on_train_only': True, 'standardization_mean': pipeline['scale'].mean_.tolist(),
              'standardization_scale': pipeline['scale'].scale_.tolist(), 'coef': pipeline['logistic'].coef_.tolist(),
              'intercept': pipeline['logistic'].intercept_.tolist(), 'class_order': pipeline['logistic'].classes_.tolist(),
              'hyperparameters_tuned': False, 'validation_or_test_used_to_fit': False},
              'interpretation_limits': ['Post-hoc analysis prompted by reviewer after external CNN results were known; not a prospective independent replication.',
              'Predictive geometry/brightness would demonstrate label association but not prove CNN reliance on these cues.',
              'Low predictive performance would not exclude other visual, acquisition or cropping shortcuts.',
              'Prepared-PNG geometry and brightness cannot identify DICOM acquisition, windowing, referral criteria, or raw-image processing.',
              'Class size imbalance requires baseline/balanced-accuracy context; 25% is not an adequate sole comparator.',
              'CNN, cohort labels, splits, manuscript, public repository and primary model results were not modified.']}
    write_json('geometry_brightness_results.json', result)
    ext = split_results['test']['logistic']
    base = split_results['test']['train_majority_baseline']
    ci = ext['accuracy_ci95_wilson']
    sig = [r['feature'] for r in kw_rows if r['split'] == 'test' and r['p_holm_within_split_four_features'] < .05]
    lines = ['EKSPLORACYJNA ANALIZA WYMIARÓW PNG I ŚREDNIEJ JASNOŚCI',
             'Analiza post hoc po poznaniu wyników głównego eksperymentu; nie była prerejestrowana. Plan tej analizy zapisano przed jej wykonaniem.', '',
             'Dane: 824 oryginalne przygotowane PNG, wszystkie sumy bajtowe zgodne z manifestem eksperymentu. Zachowano train 579, val 95, external test 150.',
             'Cechy: szerokość, wysokość, iloraz szerokości/wysokości oraz średnia jasność RGB/255 ze wszystkich pikseli. Nie są to parametry akwizycji DICOM.',
             'Model: standaryzacja wyłącznie na train i wieloklasowa regresja logistyczna (C=1, lbfgs, max_iter=5000). Bez strojenia, wyboru konfiguracji lub zmiany podziałów.', '',
             f"Zewnętrzny test: {ext['correct']}/150 poprawnych, accuracy {100*ext['accuracy']:.2f}% (95% CI Wilsona {100*ci[0]:.2f}–{100*ci[1]:.2f}%), balanced accuracy {100*ext['balanced_accuracy']:.2f}%, macro F1 {ext['macro_f1']:.4f}.",
             f"Baseline: zawsze klasa najczęstsza w train ({NAMES[majority_label]}): {base['correct']}/150, accuracy {100*base['accuracy']:.2f}%, balanced accuracy {100*base['balanced_accuracy']:.2f}%.",
             f"Walidacja wewnętrzna: accuracy {100*split_results['val']['logistic']['accuracy']:.2f}%; train: {100*split_results['train']['logistic']['accuracy']:.2f}%.",
             'Wartość 25% dla czterech jednakowo losowanych klas służy wyłącznie orientacji; nie jest wystarczającym punktem odniesienia dla nierównych klas.', '',
             'Testy Kruskala–Wallisa wykonano osobno w każdym podzbiorze i zastosowano korektę Holma dla czterech cech w danym podzbiorze.',
             'Cechy z p_Holm < 0,05 w zbiorze zewnętrznym: ' + (', '.join(sig) if sig else 'brak') + '.', '',
             'INTERPRETACJA',
             'Zależność samych wymiarów/jasności od etykiet może wskazywać na potencjalne cechy zakłócające. Nie dowodzi, że YOLO11 faktycznie korzystał z tych cech.',
             'Wynik bliski baseline nie wyklucza innych skrótów, np. sygnałów tekstury, tła, aparatu lub sposobu kadrowania. Ta analiza nie „zamyka tematu” shortcut learning.',
             'Nie trenowano ponownie modeli obrazowych i nie zmieniono artykułu, publicznych materiałów, etykiet ani wyników głównych.', '',
             'PLIKI',
             'analysis_plan_frozen_before_execution.json — plan i wersje bibliotek;',
             'image_geometry_brightness_features.csv — cechy wszystkich obrazów;',
             'class_feature_descriptives.csv — n, mediana, Q1 i Q3 według klasy i podzbioru;',
             'kruskal_wallis_holm.csv — H, surowe p i p po korekcie;',
             'logistic_predictions.csv — predykcje i prawdopodobieństwa;',
             'geometry_brightness_results.json — wyniki oraz współczynniki umożliwiające odtworzenie predykcji;',
             'analyze_geometry_brightness.py — kompletny kod analizy.']
    (OUT / 'PODSUMOWANIE_ANALIZY.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'external': ext, 'baseline': base, 'validation': split_results['val']['logistic'],
                      'external_kw_significant_features': sig, 'n_iter': result['model']['n_iter']}, indent=2))


if __name__ == '__main__':
    main()
