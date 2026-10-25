"""Metadata/compatibility guard tests only: no fitting, image reads or inference."""
import ast
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import study_design as sd
import train_models
import evaluate_models

ROOT = Path(__file__).resolve().parent.parent


class StudyDesignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original_plan = ROOT / 'runs/training_plan.json'

    def tearDown(self):
        self.temp.cleanup()

    def new_plan(self):
        run = self.root / 'runs'
        run.mkdir(exist_ok=True)
        manifest = json.loads((ROOT/'runs/development_manifest.json').read_text())
        plan = train_models.create_or_check_plan(run, manifest, ROOT/'data/clean_manifest.csv')
        return run, plan

    def test_exact_original_is_corrected_without_mutation(self):
        before = sd.sha256(self.original_plan)
        value = sd.validate_plan_design(self.original_plan)
        self.assertEqual(value['study_design_scope'], 'original_study')
        self.assertIs(value['external_test_previously_examined'], False)
        self.assertIs(value['source_external_test_previously_examined'], True)
        self.assertEqual(before, sd.sha256(self.original_plan))
        self.assertEqual(sd.interpret_evaluation(self.original_plan, ROOT/'analysis/analysis_complete.json'), value)

    def test_unknown_legacy_and_unknown_design_rejected(self):
        plan = json.loads(self.original_plan.read_text())
        path = self.root/'plan.json'
        path.write_text(json.dumps(plan))  # same values but not the registered source bytes
        with self.assertRaises(ValueError): sd.validate_plan_design(path)
        design = json.loads((ROOT/'study_design.json').read_text())
        design['source'] += ' edited'
        path.write_text(json.dumps(design))
        with self.assertRaises(ValueError): sd.load_design(path)

    def test_new_plan_is_scoped_reproduction(self):
        run, plan = self.new_plan()
        value = sd.validate_plan_design(run/'training_plan.json')
        self.assertEqual(value['study_design_scope'], 'future_reproduction')
        self.assertIs(value['external_test_previously_examined'], True)
        self.assertEqual(plan['configs']['ultralytics'], train_models.YOLO_CONFIG)
        self.assertEqual(plan['configs']['torchvision'], train_models.TV_CONFIG)
        self.assertEqual(plan['reference_seed'], 0)
        self.assertEqual(plan['clean_manifest_sha256'], sd.sha256(ROOT/'data/clean_manifest.csv'))
        # Repeated metadata freeze is idempotent, not a new plan.
        _, repeated = self.new_plan()
        self.assertEqual(plan, repeated)

    def test_tampered_active_plan_or_helper_binding_rejected(self):
        run, plan = self.new_plan()
        plan['reference_seed'] = 1
        (run/'training_plan.json').write_text(json.dumps(plan))
        with self.assertRaises(ValueError): sd.validate_plan_design(run/'training_plan.json')
        plan['reference_seed'] = 0
        plan['study_design_helper_sha256'] = '0'*64
        plan['protocol_sha256'] = sd.canonical_hash({k:v for k,v in plan.items() if k not in ('protocol_sha256','created_utc')})
        (run/'training_plan.json').write_text(json.dumps(plan))
        with self.assertRaises(ValueError): sd.validate_plan_design(run/'training_plan.json')

    def test_original_code_root_required_for_original_checkpoints(self):
        sd.validate_frozen_code_root(self.original_plan, ROOT/'provenance/original_code')
        with self.assertRaises(ValueError): sd.validate_frozen_code_root(self.original_plan, ROOT/'code')

    def test_original_lock_allows_only_relocation_not_hash_changes(self):
        path = ROOT/'provenance/original_external_evaluation_started.json'
        previous = json.loads(path.read_text())
        request = copy.deepcopy(previous)
        request.update(sd.reproduction_metadata())
        for row in request['records']: row['checkpoint']='/relocated/'+Path(row['checkpoint']).name
        sd.validate_evaluation_lock(path, request)
        request['records'][0]['checkpoint_sha256'] = '0'*64
        with self.assertRaises(ValueError): sd.validate_evaluation_lock(path, request)
        altered = self.root/'lock.json'
        altered.write_text(json.dumps(previous))
        with self.assertRaises(ValueError): sd.validate_evaluation_lock(altered, request)

    def test_full_freeze_still_checks_all_checkpoint_hashes(self):
        run, plan = self.new_plan()
        # Tiny synthetic bytes test only hash guards; these are never loaded as models.
        for model in plan['models']:
            for seed in plan['seeds']:
                dest=run/model/f'seed_{seed}'; dest.mkdir(parents=True)
                checkpoint=dest/'synthetic.bin'; checkpoint.write_bytes(f'{model}/{seed}'.encode())
                meta={'status':'complete','model':model,'seed':seed,'protocol_sha256':plan['protocol_sha256'],
                      'checkpoint':'synthetic.bin','checkpoint_sha256':sd.sha256(checkpoint),
                      'class_names':evaluate_models.CLASSES,'backend':'synthetic_guard_test','preprocess':{}}
                (dest/'run_complete.json').write_text(json.dumps(meta))
        _, freeze=evaluate_models.validate_training_freeze(run)
        self.assertEqual(freeze['n_complete_runs'],20)
        self.assertEqual(freeze['study_design_scope'],'future_reproduction')
        (run/plan['models'][-1]/'seed_4/synthetic.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'Checkpoint checksum mismatch'):
            evaluate_models.validate_training_freeze(run)

    def test_scientific_function_bodies_unchanged(self):
        allowed={'train_models.py':{'create_or_check_plan'},
                 'evaluate_models.py':{'validate_training_freeze','main'},
                 'eigen_cam.py':{'load_locked_artifacts','main'},
                 'eigen_cam_replay.py':{'load_locked_artifacts','main'}}
        for name, permitted in allowed.items():
            def funcs(path):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
            old=funcs(ROOT/'provenance/original_code'/name);new=funcs(ROOT/'code'/name)
            self.assertEqual(set(old),set(new))
            for function in old:
                if function not in permitted: self.assertEqual(old[function],new[function],f'{name}:{function}')

if __name__=='__main__': unittest.main()
