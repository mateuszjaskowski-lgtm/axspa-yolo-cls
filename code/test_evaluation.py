"""CPU checks of statistical definitions and paired inference (synthetic data)."""
import json
import unittest

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

import evaluate_models as e


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        cm = np.array([[51,4,0,0],[3,49,1,1],[0,4,16,0],[1,1,0,19]])
        truth, prediction = [], []
        for i in range(4):
            for j in range(4):
                truth.extend([i] * int(cm[i,j])); prediction.extend([j] * int(cm[i,j]))
        self.y = np.array(truth); self.prediction = np.array(prediction)
        self.p = np.full((len(truth),4), .02)
        self.p[np.arange(len(truth)),self.prediction] = .94

    def test_library_agreement_and_brier_conventions(self):
        scores = e.basic_scores(self.y,self.p)
        self.assertEqual(scores['accuracy'], .9)
        self.assertAlmostEqual(scores['unweighted_cohen_kappa'], cohen_kappa_score(self.y,self.prediction))
        self.assertAlmostEqual(scores['balanced_accuracy'], balanced_accuracy_score(self.y,self.prediction))
        self.assertAlmostEqual(scores['macro_f1'], f1_score(self.y,self.prediction,average='macro'))
        self.assertAlmostEqual(scores['macro_ovr_auc'], roc_auc_score(self.y,self.p,multi_class='ovr',average='macro'))
        self.assertAlmostEqual(scores['multiclass_brier'], 4*scores['mean_ovr_brier'])

    def test_class_denominators_and_exact_intervals(self):
        threshold = e.choose_val_threshold(self.y,self.p)
        result = e.evaluate(self.y,self.p,threshold,20,27)
        normal, osteophytes = result['per_class'][:2]
        self.assertEqual(normal['sensitivity']['numerator'],51)
        self.assertEqual(normal['sensitivity']['denominator'],55)
        self.assertEqual(osteophytes['ppv']['denominator'],58)
        ci = e.exact_binomial(135,150)
        expected = binomtest(135,150).proportion_ci(method='exact')
        self.assertAlmostEqual(ci['ci_low'],expected.low)
        self.assertAlmostEqual(ci['ci_high'],expected.high)
        self.assertEqual(e.exact_binomial(0,12)['ci_low'],0)
        self.assertEqual(e.exact_binomial(12,12)['ci_high'],1)
        self.assertIsNone(e.exact_binomial(0,0)['estimate'])
        self.assertEqual(len(result['calibration']),6)

    def test_undefined_and_degenerate_bootstraps(self):
        result = e.evaluate(np.array([0,0]),np.array([[1,0,0,0],[1,0,0,0.]]),{'threshold':.5},20)
        self.assertEqual(result['metrics']['macro_ovr_auc']['valid_replicates'],0)
        self.assertEqual(result['metrics']['unweighted_cohen_kappa']['valid_replicates'],0)
        self.assertTrue(result['metrics']['multiclass_brier']['degenerate_empirical_distribution'])
        json.dumps(e.clean_json(result),allow_nan=False)

    def test_pairing_and_multiplicity(self):
        identical = e.paired_comparison(self.y,self.p,self.p,20)
        self.assertEqual(identical['mcnemar_exact_two_sided_p'],1)
        self.assertEqual(identical['discordant_pairs'],0)
        self.assertEqual(identical['differences']['accuracy']['estimate_reference_minus_comparator'],0)
        self.assertEqual(e.holm_adjust([.01,.04,.03]),[.03,.06,.06])
        other = self.p.copy()
        # Force three previously correct normal predictions to be wrong.
        other[:3] = [.02,.94,.02,.02]
        paired = e.paired_comparison(self.y,self.p,other,20)
        self.assertEqual(paired['reference_correct_comparator_wrong'],3)
        self.assertEqual(paired['reference_wrong_comparator_correct'],0)
        self.assertEqual(paired['mcnemar_exact_two_sided_p'],.25)

    def test_threshold_rule_is_deterministic(self):
        # Constant probabilities give every operating point Youden=0;
        # choose the highest finite threshold, an always-negative decision.
        y = np.array([0,1,2,3]); p = np.full((4,4),.25)
        info = e.choose_val_threshold(y,p)
        self.assertGreater(info['threshold'],.5)
        self.assertEqual(info['max_youden'],0)
        self.assertEqual(info['tied_candidate_count'],2)


if __name__ == '__main__':
    unittest.main()
