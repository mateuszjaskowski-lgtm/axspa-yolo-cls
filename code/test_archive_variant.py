"""Strict byte-variant registration tests; no training or model inference.

Set AXSPA_EQUIVALENT_ARCHIVE to the private, registered Baza docelowa.zip to
include the 824-image integration checks. No source images are distributed here.
"""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_data as prep
import study_design as sd
import train_models

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_ARCHIVE = os.environ.get('AXSPA_EQUIVALENT_ARCHIVE')


class ArchiveVariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_plan(self, digest, scope='future_reproduction'):
        run = self.root / 'runs'
        run.mkdir(exist_ok=True)
        development = json.loads((ROOT/'runs/development_manifest.json').read_text())
        plan = train_models.create_or_check_plan(run, development, ROOT/'data/clean_manifest.csv')
        plan['clean_manifest_sha256'] = digest
        plan['study_design_scope'] = scope
        plan['protocol_sha256'] = sd.canonical_hash({k: v for k, v in plan.items()
                                                   if k not in ('protocol_sha256', 'created_utc')})
        path = run/'training_plan.json'
        path.write_text(json.dumps(plan))
        return path

    def test_reference_and_registered_hashes_are_exact(self):
        self.assertEqual((ROOT/'code/reference_clean_manifest.csv').read_bytes(),
                         (ROOT/'data/clean_manifest.csv').read_bytes())
        self.assertEqual(prep.sha_file(ROOT/'code/reference_clean_manifest.csv'), prep.CLEAN_MANIFEST_SHA256)
        self.assertEqual(prep.PNG_BYTE_VARIANT_MANIFEST_SHA256, sd.PNG_BYTE_VARIANT_MANIFEST_SHA256)
        self.assertEqual(len(prep.ACCEPTED_ARCHIVES), 3)
        # Recover the old output layout without changing any historical values.
        with (ROOT/'code/original_manifest.csv').open(newline='') as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
        with (ROOT/'data/clean_manifest.csv').open(newline='') as f:
            reference = {r['relative_path']: r for r in csv.DictReader(f)}
        omitted = {drop for _, drop in prep.PAIRS}
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields+['independent_rgb_sha256'])
        writer.writeheader()
        writer.writerows({**r, 'independent_rgb_sha256': reference[r['relative_path']]['independent_rgb_sha256']}
                         for r in rows if r['relative_path'] not in omitted)
        self.assertEqual(stream.getvalue().encode(), (ROOT/'data/clean_manifest.csv').read_bytes())

    def test_registered_variant_is_future_reproduction_only(self):
        path = self.make_plan(prep.PNG_BYTE_VARIANT_MANIFEST_SHA256)
        self.assertEqual(sd.validate_plan_design(path)['study_design_scope'], 'future_reproduction')
        plan = json.loads(path.read_text())
        plan['study_design_scope'] = 'original_study'
        plan['protocol_sha256'] = sd.canonical_hash({k:v for k,v in plan.items()
                                                   if k not in ('protocol_sha256','created_utc')})
        path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            sd.validate_plan_design(path)

    def test_unknown_reproduction_manifest_rejected(self):
        path = self.make_plan('0'*64)
        with self.assertRaisesRegex(ValueError, 'explicitly registered manifest'):
            sd.validate_plan_design(path)

    def test_exact_original_study_binding_unchanged(self):
        path = ROOT/'runs/training_plan.json'
        before = sd.sha256(path)
        self.assertEqual(sd.validate_plan_design(path)['study_design_scope'], 'original_study')
        self.assertEqual(json.loads(path.read_text())['clean_manifest_sha256'], prep.CLEAN_MANIFEST_SHA256)
        self.assertEqual(before, sd.sha256(path))

    def test_unregistered_archive_rejected_before_output(self):
        archive = self.root/'unknown.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('arbitrary.txt', 'not registered')
        output = self.root/'output'
        with self.assertRaisesRegex(ValueError, 'three explicitly registered'):
            prep.main(['--archive', str(archive), '--output-dir', str(output)])
        self.assertFalse(output.exists())

    @unittest.skipUnless(PRIVATE_ARCHIVE, 'Set AXSPA_EQUIVALENT_ARCHIVE for private 824-image checks')
    def test_registered_824_image_archive_prepares_actual_hash_manifest(self):
        archive = Path(PRIVATE_ARCHIVE)
        output = self.root/'prepared'
        with contextlib.redirect_stdout(io.StringIO()):
            prep.main(['--archive', str(archive), '--output-dir', str(output)])
        self.assertEqual(prep.sha_file(output/'clean_manifest.csv'), prep.PNG_BYTE_VARIANT_MANIFEST_SHA256)
        with (output/'clean_manifest.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 824)
        for row in rows:
            self.assertEqual(prep.sha_file(output/'dataset'/row['relative_path']), row['sha256'])
        audit = json.loads((output/'data_audit.json').read_text())
        self.assertTrue(audit['png_byte_variant']['rgb_pixels_dimensions_paths_and_labels_match_reference'])
        self.assertEqual(len(audit['png_byte_variant']['file_byte_hash_mapping']), 824)
        self.assertEqual(audit['removed_duplicate_copies_this_invocation'], [])
        before = (output/'data_audit.json').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            prep.main(['--archive', str(archive), '--output-dir', str(output)])
        self.assertEqual((output/'data_audit.json').read_bytes(), before)
        self.assertEqual(len(list((output/'verification_invocations').glob('*.json'))), 1)

    @unittest.skipUnless(PRIVATE_ARCHIVE, 'Set AXSPA_EQUIVALENT_ARCHIVE for private archive checks')
    def test_changed_archive_bytes_are_rejected(self):
        altered = self.root/'altered.zip'
        altered.write_bytes(Path(PRIVATE_ARCHIVE).read_bytes()+b'changed archive bytes')
        with self.assertRaisesRegex(ValueError, 'three explicitly registered'):
            prep.main(['--archive', str(altered), '--output-dir', str(self.root/'output')])

    @unittest.skipUnless(PRIVATE_ARCHIVE, 'Set AXSPA_EQUIVALENT_ARCHIVE for private archive checks')
    def test_original_archive_branch_still_requires_original_png_bytes(self):
        # Exercise the OLD branch with the current differing PNG bytes. Only
        # the outer archive identity is mocked in this test; image hashes remain
        # real, and the original per-PNG byte guard must reject the first file.
        archive = Path(PRIVATE_ARCHIVE)
        real_sha = prep.sha_file
        def test_archive_identity(path):
            return prep.ALREADY_CLEAN_ARCHIVE_SHA256 if Path(path) == archive else real_sha(path)
        with patch.object(prep, 'sha_file', side_effect=test_archive_identity):
            with self.assertRaisesRegex(ValueError, 'Image-byte mismatch'):
                prep.main(['--archive', str(archive), '--output-dir', str(self.root/'output')])


if __name__ == '__main__':
    unittest.main()
