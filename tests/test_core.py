"""Synthetic, independently checked regression tests for the validation engine."""
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from cmvs.core import read_csv, validate


def fixture(n=160):
    rng = np.random.default_rng(124)
    x = rng.normal(size=n)
    outcome = np.where(x + rng.normal(size=n) > 0, 'event', 'control')
    return pd.DataFrame({'outcome': outcome, 'age': 50 + 12*x,
                         'marker': rng.normal(size=n),
                         'site': np.where(np.arange(n) % 3 == 0, 'A', 'B')})


def evaluate(data=None, **kwargs):
    options = dict(test_size=.25, seed=31, bootstrap=0)
    options.update(kwargs)
    return validate(fixture() if data is None else data, 'outcome', 'event',
                    ['age', 'marker'], ['site'], **options)


class InputTests(unittest.TestCase):
    def test_csv_preserves_text_codes_and_bom(self):
        data = read_csv(io.BytesIO('\ufeffoutcome,code\n01,001\n1,1\n'.encode('utf-8')))
        self.assertEqual(list(data['outcome']), ['01', '1'])
        self.assertEqual(list(data['code']), ['001', '1'])

    def test_csv_duplicate_headers_rejected(self):
        with self.assertRaises(ValueError):
            read_csv(io.BytesIO(b'outcome,x,x\na,1,2\nb,2,3\n'))

    def test_csv_ragged_rows_rejected(self):
        for content in (b'outcome,x\na,1,2\nb,2\n', b'outcome,x\na\nb,2\n'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_csv(io.BytesIO(content))

    def test_text_event_labels_are_not_numerically_collapsed(self):
        data = fixture()
        data['outcome'] = data['outcome'].map({'event': '01', 'control': '1'})
        result = validate(data, 'outcome', '01', ['age'], ['site'], seed=31)
        for row in result['predictions']:
            self.assertEqual(row['observed'], int(data.iloc[row['row']]['outcome'] == '01'))

    def test_missing_outcome_exclusion(self):
        data = fixture()
        missing = ['', 'NA', 'n/a', 'NULL', 'NaN', None]
        for i, value in enumerate(missing):
            data.loc[i, 'outcome'] = value
        result = evaluate(data)
        audit = result['audit']
        self.assertEqual(audit['n_missing_outcome'], len(missing))
        self.assertEqual(audit['n_train'] + audit['n_test'], len(data)-len(missing))
        self.assertFalse(set(range(len(missing))) & set(audit['train_rows'] + audit['test_rows']))

    def test_invalid_event_and_multiclass_rejected(self):
        data = fixture()
        with self.assertRaises(ValueError):
            validate(data, 'outcome', 'absent', ['age'], [])
        data.loc[0, 'outcome'] = 'third'
        with self.assertRaises(ValueError):
            evaluate(data)

    def test_outcome_predictor_and_unknown_columns_rejected(self):
        for predictors in (['outcome'], ['missing_column']):
            with self.subTest(predictors=predictors), self.assertRaises(ValueError):
                validate(fixture(), 'outcome', 'event', predictors, [])

    def test_degenerate_requests_fail_clearly(self):
        for value in (0, 1, -.2):
            with self.subTest(test_size=value), self.assertRaises(ValueError):
                evaluate(test_size=value)
        data = fixture()
        data['outcome'] = 'event'
        with self.assertRaises(ValueError):
            evaluate(data)
        with self.assertRaises(ValueError):
            validate(fixture(), 'outcome', 'event', [], [])


    def test_too_small_minority_and_no_usable_features_rejected(self):
        data = fixture()
        data['outcome'] = 'control'
        data.loc[:2, 'outcome'] = 'event'
        with self.assertRaises(ValueError):
            evaluate(data)
        data = fixture()
        data['constant'] = 1
        with self.assertRaises(ValueError):
            validate(data, 'outcome', 'event', ['constant'], [])

    def test_duplicate_or_overlapping_predictors_rejected(self):
        for numeric, categorical in ((['age', 'age'], []), (['age'], ['age'])):
            with self.subTest(numeric=numeric, categorical=categorical), self.assertRaises(ValueError):
                validate(fixture(), 'outcome', 'event', numeric, categorical)


class LeakageTests(unittest.TestCase):
    def test_holdout_mutation_cannot_change_training_preprocessing(self):
        data = fixture()
        baseline = evaluate(data)
        audit = baseline['audit']
        changed = data.copy()
        changed.loc[audit['test_rows'], 'age'] = 1e6
        changed.loc[audit['test_rows'], 'marker'] = np.nan
        changed.loc[audit['test_rows'], 'site'] = 'HELDOUT_ONLY_CATEGORY'
        result = evaluate(changed)
        self.assertEqual(result['audit']['train_rows'], audit['train_rows'])
        self.assertEqual(result['audit']['test_rows'], audit['test_rows'])
        for name, original in audit['preprocessing'].items():
            for key in ('kind', 'median', 'mean', 'scale', 'mode', 'categories'):
                if key in original:
                    self.assertEqual(result['audit']['preprocessing'][name][key], original[key])
        self.assertEqual(result['audit']['preprocessing']['site']['unseen_holdout'], audit['n_test'])
        self.assertEqual(result['audit']['preprocessing']['marker']['missing'], audit['n_test'])
        self.assertNotIn('HELDOUT_ONLY_CATEGORY', result['audit']['preprocessing']['site']['categories'])
        self.assertTrue(all(np.isfinite(p['probability']) for p in result['predictions']))

    def test_model_fit_never_sees_holdout_rows(self):
        data = fixture()
        fitted = []
        original_fit = LogisticRegression.fit
        def capture_fit(model, x, y, *args, **kwargs):
            fitted.append((np.array(x, copy=True), np.array(y, copy=True)))
            return original_fit(model, x, y, *args, **kwargs)
        with patch('cmvs.core.LogisticRegression.fit', new=capture_fit):
            first = evaluate(data)
            changed = data.copy()
            changed.loc[first['audit']['test_rows'], 'age'] = 1e6
            changed.loc[first['audit']['test_rows'], 'site'] = 'never_in_training'
            evaluate(changed)
        self.assertEqual(len(fitted), 2)
        self.assertEqual(len(fitted[0][0]), first['audit']['n_train'])
        np.testing.assert_array_equal(fitted[0][0], fitted[1][0])
        np.testing.assert_array_equal(fitted[0][1], fitted[1][1])

    def test_training_median_matches_independent_calculation(self):
        data = fixture()
        data.loc[::4, 'age'] = np.nan
        result = evaluate(data)
        rows = result['audit']['train_rows']
        self.assertAlmostEqual(result['audit']['preprocessing']['age']['median'],
                               float(data.loc[rows, 'age'].median()))
        train = set(rows)
        test = set(result['audit']['test_rows'])
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(range(len(data))))

    def test_constant_and_missing_features_are_audited(self):
        data = fixture()
        data['constant'] = 3
        data['empty'] = np.nan
        result = validate(data, 'outcome', 'event', ['age', 'constant', 'empty'], ['site'])
        dropped = str(result['audit']['dropped_features'])
        self.assertIn('constant', dropped)
        self.assertIn('empty', dropped)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.result = evaluate(cls.data)
        cls.y = np.array([p['observed'] for p in cls.result['predictions']])
        cls.p = np.array([p['probability'] for p in cls.result['predictions']])

    def test_metrics_against_sklearn(self):
        metrics = self.result['metrics']
        expected = {'roc_auc': roc_auc_score(self.y, self.p),
                    'average_precision': average_precision_score(self.y, self.p),
                    'brier_score': brier_score_loss(self.y, self.p),
                    'log_loss': log_loss(self.y, self.p)}
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(metrics[key], value, places=11)

    def test_comparator_uses_training_prevalence(self):
        prevalence = float((self.data.iloc[self.result['audit']['train_rows']]['outcome'] == 'event').mean())
        comparator = self.result['comparator']
        self.assertAlmostEqual(comparator['probability'], prevalence)
        p = np.full(len(self.y), prevalence)
        self.assertAlmostEqual(comparator['brier_score'], brier_score_loss(self.y, p))
        self.assertAlmostEqual(comparator['roc_auc'], .5)

    def test_decision_curve_against_direct_counts(self):
        rows = self.result['decision_curve']
        self.assertEqual(len(rows), 19)
        for row in rows:
            threshold = row['threshold']
            self.assertGreater(threshold, 0)
            self.assertLess(threshold, 1)
            positive = self.p >= threshold
            tp = int(np.sum(positive & (self.y == 1)))
            fp = int(np.sum(positive & (self.y == 0)))
            expected = (tp - fp * threshold/(1-threshold))/len(self.y)
            self.assertAlmostEqual(row['net_benefit'], expected, places=12)
            self.assertAlmostEqual(row['treat_all'], self.y.mean()-(1-self.y.mean())*threshold/(1-threshold))
            self.assertEqual(row['treat_none'], 0)

    def test_calibration_bins_account_for_all_holdout_rows(self):
        rows = self.result['calibration']
        self.assertEqual(sum(row['n'] for row in rows), len(self.y))
        self.assertAlmostEqual(sum(row['n']*row['observed_rate'] for row in rows)/len(self.y), self.y.mean())
        self.assertAlmostEqual(sum(row['n']*row['mean_predicted'] for row in rows)/len(self.y), self.p.mean())

    def test_bootstrap_reproducibility_and_interval_integrity(self):
        first = evaluate(self.data, bootstrap=150)
        second = evaluate(self.data, bootstrap=150)
        self.assertEqual(first['bootstrap'], second['bootstrap'])
        result = first['bootstrap']
        self.assertEqual(result['requested'], 150)
        self.assertEqual(result['valid'] + result['skipped'], 150)
        self.assertGreaterEqual(result['valid'], 100)
        self.assertEqual(set(result['intervals']), {'roc_auc', 'average_precision', 'brier_score', 'log_loss'})
        for name, interval in result['intervals'].items():
            with self.subTest(metric=name):
                self.assertTrue(np.isfinite(interval['lower']))
                self.assertTrue(np.isfinite(interval['upper']))
                self.assertLessEqual(interval['lower'], interval['upper'])
                self.assertGreaterEqual(interval['lower'], 0)
                if name != 'log_loss':
                    self.assertLessEqual(interval['upper'], 1)
        self.assertEqual(first['predictions'], self.result['predictions'])
        json.dumps(first, allow_nan=False)

    def test_deterministic_and_strict_json_serializable(self):
        other = evaluate(self.data)
        self.assertEqual(self.result, other)
        json.dumps(self.result, allow_nan=False)
        self.assertTrue(np.all((self.p >= 0) & (self.p <= 1)))


class FixedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'outcome': ['0', '0', '1', '1', '0', '1', '0', '1'],
                                  'risk': [.1, .4, .3, .8, .2, .7, .6, .9]})

    def test_known_probabilities_hand_calculated_metrics(self):
        from cmvs.core import evaluate_predictions
        original = self.data.copy(deep=True)
        with patch('cmvs.core.LogisticRegression.fit', side_effect=AssertionError('Must not refit')):
            result = evaluate_predictions(self.data, 'outcome', '1', 'risk')
        self.assertAlmostEqual(result['metrics']['roc_auc'], .875)
        self.assertAlmostEqual(result['metrics']['brier_score'], .15)
        self.assertIsNone(result['comparator'])
        midpoint = next(row for row in result['decision_curve'] if abs(row['threshold']-.5) < 1e-10)
        self.assertAlmostEqual(midpoint['net_benefit'], .25)
        self.assertEqual(len(result['predictions']), len(self.data))
        pd.testing.assert_frame_equal(original, self.data)
        json.dumps(result, allow_nan=False)

    def test_bad_probabilities_rejected_not_silently_dropped(self):
        from cmvs.core import evaluate_predictions
        for value in (-.1, 1.1, np.inf, np.nan, 'invalid'):
            data = self.data.copy()
            data['risk'] = data['risk'].astype(object)
            data.loc[0, 'risk'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_predictions(data, 'outcome', '1', 'risk')

    def test_small_bootstrap_suppresses_unstable_intervals(self):
        from cmvs.core import evaluate_predictions
        result = evaluate_predictions(self.data, 'outcome', '1', 'risk', bootstrap=30)
        self.assertEqual(result['bootstrap']['requested'], 30)
        self.assertEqual(result['bootstrap']['intervals'], {})
        self.assertEqual(result['bootstrap']['valid'] + result['bootstrap']['skipped'], 30)


if __name__ == '__main__':
    unittest.main()
