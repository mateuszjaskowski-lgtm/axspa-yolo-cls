"""Scoped author correction; never rewrite original frozen artifacts or relax hashes.

The original-study statement is author-reported. Every future rerun uses the
separate reproduction scope. This helper does not fit models or read image data.
"""
from pathlib import Path
import hashlib
import json

EXPECTED_DESIGN_SHA256 = '121c944bb74d7f9fbf353b388a47df99f8003fe3c872e29bfae2555ffe88bb6f'
DEFAULT_DESIGN = Path(__file__).resolve().parent.parent / 'study_design.json'
# Exact manifest of the independently verified 824-image PNG byte variant.
# Permitted only for future reproduction; original-study bindings stay unchanged.
PNG_BYTE_VARIANT_MANIFEST_SHA256 = 'f98cc4992fe3da8140ed1ab033361e86188fc3e9c9c04d814ac09af3bb869642'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def load_design(path=None):
    path = Path(path or DEFAULT_DESIGN)
    if sha256(path) != EXPECTED_DESIGN_SHA256:
        raise ValueError('Unregistered author-confirmed study-design file')
    design = json.loads(path.read_text(encoding='utf-8'))
    if design.get('schema_version') != 1:
        raise ValueError('Unsupported study-design schema')
    if (design['original_study']['external_test_previously_examined'] is not False
            or design['future_reproduction']['external_test_previously_examined'] is not True
            or design['future_reproduction']['claim_of_new_untouched_external_validation'] is not False):
        raise ValueError('Original study and future reproduction scopes must remain distinct')
    return design, sha256(path)


def reproduction_metadata(path=None):
    design, digest = load_design(path)
    return {'study_design_sha256': digest, 'study_design_scope': 'future_reproduction',
            'split_method': design['split']['method'],
            'external_test_previously_examined': design['future_reproduction']['external_test_previously_examined'],
            'evaluation_role': design['future_reproduction']['evaluation_role']}


def validate_plan_design(plan_path, design_path=None):
    """Accept only the exact registered original plan or a bound active plan."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    design, digest = load_design(design_path)
    protocol = {k: v for k, v in plan.items() if k not in ('protocol_sha256', 'created_utc')}
    if canonical_hash(protocol) != plan.get('protocol_sha256'):
        raise ValueError('Training-plan canonical protocol hash mismatch')
    if plan.get('reference_seed') != 0 or plan.get('primary_model') != 'yolo11l-cls':
        raise ValueError('Primary model and seed must remain fixed')
    legacy = design['legacy_registration']
    if sha256(plan_path) == legacy['training_plan_sha256']:
        if plan.get('clean_manifest_sha256') != design['split']['clean_manifest_sha256']:
            raise ValueError('Original study must retain its historical manifest')
        if plan['protocol_sha256'] != legacy['protocol_sha256']:
            raise ValueError('Registered original protocol differs')
        return {'study_design_sha256': digest, 'study_design_scope': 'original_study',
                'split_method': design['split']['method'],
                'external_test_previously_examined': design['original_study']['external_test_previously_examined'],
                'source_external_test_previously_examined': plan.get('external_test_previously_examined'),
                'interpretation_source': 'Hash-bound author correction; original file is unchanged'}
    expected = reproduction_metadata(design_path)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Unregistered legacy plan or mismatched active study-design binding')
    if plan.get('clean_manifest_sha256') not in {
            design['split']['clean_manifest_sha256'], PNG_BYTE_VARIANT_MANIFEST_SHA256}:
        raise ValueError('Reproduction plan does not reference an explicitly registered manifest')
    if plan.get('study_design_helper_sha256') != sha256(__file__):
        raise ValueError('Study-design helper differs from the new frozen protocol')
    return expected


def interpret_evaluation(plan_path, complete_path, design_path=None):
    plan_scope = validate_plan_design(plan_path, design_path)
    complete_path = Path(complete_path)
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    if complete.get('training_plan_sha256') != sha256(plan_path):
        raise ValueError('Completed evaluation uses a different training plan')
    design, _ = load_design(design_path)
    expected_hash = design['legacy_registration']['artifact_sha256']['analysis/analysis_complete.json']
    if sha256(complete_path) == expected_hash:
        if plan_scope['study_design_scope'] != 'original_study':
            raise ValueError('Original evaluation cannot be assigned to a different plan')
        return plan_scope
    expected = reproduction_metadata(design_path)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError('Unknown evaluation metadata: no hash-bound original or explicit reproduction scope')
    return expected


def validate_evaluation_lock(lock_path, requested_lock, design_path=None):
    """Relocating original checkpoints is allowed; identities/hashes stay exact."""
    lock_path = Path(lock_path)
    previous = json.loads(lock_path.read_text(encoding='utf-8'))
    if previous == requested_lock:
        return
    design, _ = load_design(design_path)
    legacy = design['legacy_registration']
    if (sha256(lock_path) != legacy['external_evaluation_lock_sha256']
            or requested_lock.get('training_plan_sha256') != legacy['training_plan_sha256']):
        raise ValueError('Unregistered or changed external-evaluation lock')
    for key in ('training_plan_sha256', 'clean_manifest_sha256'):
        if previous.get(key) != requested_lock.get(key):
            raise ValueError('Original lock protocol/manifest mismatch')
    def identities(records):
        # Absolute container locations are not checkpoint identities.
        return [{k: v for k, v in r.items() if k != 'checkpoint'} for r in records]
    if identities(previous['records']) != identities(requested_lock['records']):
        raise ValueError('Original locked checkpoint/run metadata changed')


def validate_frozen_code_root(plan_path, root, complete_path=None, design_path=None):
    scope = validate_plan_design(plan_path, design_path)
    design, _ = load_design(design_path)
    root = Path(root)
    if scope['study_design_scope'] == 'original_study':
        expected = design['legacy_registration']['original_code_sha256']
    else:
        plan = json.loads(Path(plan_path).read_text())
        expected = {'train_models.py': plan['runner_sha256'],
                    'spine_transforms.py': plan['transform_source_sha256'],
                    'study_design.py': plan['study_design_helper_sha256']}
        if complete_path:
            complete = json.loads(Path(complete_path).read_text())
            expected['evaluate_models.py'] = complete['evaluation_script_sha256']
    for name, digest in expected.items():
        if not (root / name).is_file() or sha256(root / name) != digest:
            raise ValueError(f'Frozen source checksum mismatch: {root / name}')
    return scope
