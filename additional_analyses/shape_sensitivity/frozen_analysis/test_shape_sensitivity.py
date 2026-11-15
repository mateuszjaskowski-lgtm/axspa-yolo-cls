"""Meaningful lightweight tests: interventions, ID alignment, replay-stop rules, statistics."""
import unittest
import numpy as np
from PIL import Image
import run_shape_sensitivity as s


class TestShapeSensitivity(unittest.TestCase):
    def test_uniform_controls_preserve_dimensions_and_half_up_mean(self):
        image = Image.fromarray(np.asarray([[[1, 1, 1], [2, 2, 2]]], dtype=np.uint8))
        constant, value = s.gray_image(image, "fixed_gray_128")
        self.assertEqual(constant.size, image.size)
        self.assertEqual(value, 128)
        self.assertTrue(np.all(np.asarray(constant) == 128))
        mean, value = s.gray_image(image, "per_image_mean_gray")
        self.assertEqual(value, 2)  # 1.5 -> 2, explicitly fixed before predictions
        self.assertEqual(mean.size, image.size)
        self.assertTrue(np.all(np.asarray(mean) == 2))
        self.assertEqual(np.asarray(image)[0, 0, 0], 1)  # source untouched

    def test_rgb_identity_includes_geometry(self):
        a = Image.new("RGB", (2, 3), "gray")
        b = Image.new("RGB", (3, 2), "gray")
        self.assertNotEqual(s.rgb_sha256(a), s.rgb_sha256(b))

    def test_prediction_alignment_by_path_not_position(self):
        manifest = [dict(relative_path="test/normal/a.png", label="0", class_name="normal"),
                    dict(relative_path="test/osteofity/b.png", label="1", class_name="osteofity")]
        rows = [dict(relative_path="osteofity/b.png", true_label=1, class_name="osteofity", probabilities=[0, 1, 0, 0]),
                dict(relative_path="normal/a.png", true_label=0, class_name="normal", probabilities=[1, 0, 0, 0])]
        np.testing.assert_array_equal(s.align_predictions(manifest, rows), [[1, 0, 0, 0], [0, 1, 0, 0]])
        rows[1]["true_label"] = 2
        with self.assertRaisesRegex(RuntimeError, "label mapping"):
            s.align_predictions(manifest, rows)

    def test_reference_alignment_rejects_identity_drift(self):
        manifest = [dict(image_id="a", sha256="hash_a", label="0"), dict(image_id="b", sha256="hash_b", label="1")]
        archive = dict(image_id=np.asarray(["b", "a"]), sha256=np.asarray(["hash_b", "hash_a"]),
                       y_true=np.asarray([1, 0]), probabilities=np.asarray([[0, 1, 0, 0], [1, 0, 0, 0]]),
                       class_names=np.asarray(s.CLASSES))
        np.testing.assert_array_equal(s.align_reference(manifest, archive), [[1, 0, 0, 0], [0, 1, 0, 0]])
        archive["sha256"][0] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "hash or label"):
            s.align_reference(manifest, archive)

    def test_replay_stops_for_probability_or_decision_drift(self):
        y = np.zeros(150, dtype=int)
        reference = np.zeros((150, 4))
        reference[:138, 0] = 1
        reference[138:, 1] = 1
        self.assertEqual(s.baseline_gate(y, reference, reference)["status"], "PASS")
        p = reference.copy()
        p[0] = [.99, .01, 0, 0]
        self.assertEqual(s.baseline_gate(y, p, reference)["status"], "FAIL")
        p[0] = [0, 1, 0, 0]
        self.assertEqual(s.baseline_gate(y, p, reference)["status"], "FAIL")

    def test_majority_baseline_not_uniform_chance(self):
        y = np.repeat(np.arange(4), [55, 54, 20, 21])
        p = np.tile([1, 0, 0, 0], (150, 1))
        result = s.metrics(y, p)
        self.assertAlmostEqual(result["accuracy"]["estimate"], 55/150)
        self.assertEqual(result["balanced_accuracy"], .25)
        self.assertAlmostEqual(result["unweighted_cohen_kappa"], 0)
        self.assertAlmostEqual(result["accuracy"]["ci_low"], .2937744456533283, places=12)

    def test_paired_bootstrap_identity_is_degenerate(self):
        y = np.repeat(np.arange(4), [55, 54, 20, 21])
        p = np.eye(4)[y]
        result = s.paired_difference(y, p, p)
        self.assertEqual(result["estimate_original_minus_variant"], 0)
        self.assertEqual(result["ci_low"], 0)
        self.assertEqual(result["ci_high"], 0)
        self.assertTrue(result["degenerate"])


if __name__ == "__main__":
    unittest.main()
