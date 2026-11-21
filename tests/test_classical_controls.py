"""
Unit tests for matched classical baseline models in experiments/phase5/code/classical_controls.py.
Verifies Extra Trees, RBF-SVM, ElasticNet, and RBF-SVR on 9-feature synthetic fixtures.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE5_CODE_DIR = os.path.join(ROOT_DIR, "experiments", "phase5", "code")
if PHASE5_CODE_DIR not in sys.path:
    sys.path.insert(0, PHASE5_CODE_DIR)

from classical_controls import (
    run_classical_classification_controls,
    run_classical_regression_controls
)


class TestClassicalControls(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Small synthetic fixture: 9 features
        self.n_tr = 30
        self.n_val = 10
        self.n_te = 10
        self.n_feat = 9

        self.X_tr = np.random.randn(self.n_tr, self.n_feat)
        self.y_tr_cls = np.random.randint(0, 2, size=self.n_tr)
        self.y_tr_reg = np.random.uniform(10.0, 35.0, size=self.n_tr)

        self.X_val = np.random.randn(self.n_val, self.n_feat)
        self.y_val_cls = np.random.randint(0, 2, size=self.n_val)
        self.y_val_reg = np.random.uniform(10.0, 35.0, size=self.n_val)

        self.X_te = np.random.randn(self.n_te, self.n_feat)
        self.y_te_cls = np.random.randint(0, 2, size=self.n_te)
        self.y_te_reg = np.random.uniform(10.0, 35.0, size=self.n_te)

        self.df_te = pd.DataFrame({
            "subject#": np.repeat([1, 2], self.n_te // 2)
        })

    def test_classification_controls_output_structure(self):
        """Verifies Extra Trees and RBF-SVM return valid probability bounds and metric dictionaries."""
        results = run_classical_classification_controls(
            self.X_tr, self.y_tr_cls,
            self.X_val, self.y_val_cls,
            self.X_te, self.y_te_cls,
            seed=42
        )

        expected_models = ["Extra Trees (9 Feats)", "Classical RBF-SVM (9 Feats)"]
        for name in expected_models:
            self.assertIn(name, results)
            res = results[name]
            self.assertIn("val_metrics", res)
            self.assertIn("test_metrics", res)
            self.assertIn("test_probs", res)

            probs = res["test_probs"]
            self.assertEqual(len(probs), self.n_te)
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0), f"{name} probs must be in [0, 1]")

            t_metrics = res["test_metrics"]
            self.assertIn("bacc", t_metrics)
            self.assertIn("fit_time", t_metrics)
            self.assertGreater(t_metrics["fit_time"], 0.0)

    def test_regression_controls_output_structure(self):
        """Verifies ElasticNet and RBF-SVR return predictions and error metrics."""
        results = run_classical_regression_controls(
            self.X_tr, self.y_tr_reg,
            self.X_val, self.y_val_reg,
            self.X_te, self.y_te_reg,
            self.df_te,
            seed=42
        )

        expected_models = ["ElasticNet (9 Feats)", "Classical RBF-SVR (9 Feats)"]
        for name in expected_models:
            self.assertIn(name, results)
            res = results[name]
            self.assertIn("val_metrics", res)
            self.assertIn("test_metrics", res)
            self.assertIn("test_preds", res)

            preds = res["test_preds"]
            self.assertEqual(len(preds), self.n_te)
            self.assertFalse(np.any(np.isnan(preds)), f"{name} predictions must not contain NaN")

            t_metrics = res["test_metrics"]
            self.assertIn("mae", t_metrics)
            self.assertIn("rmse", t_metrics)
            self.assertIn("r2", t_metrics)
            self.assertIn("mean_subj_mae", t_metrics)
            self.assertIn("fit_time", t_metrics)
            self.assertGreater(t_metrics["mae"], 0.0)

    def test_svm_hyperparameter_selection(self):
        """Verifies RBF-SVM selects C from the candidate grid {0.5, 1.0, 2.0}."""
        results = run_classical_classification_controls(
            self.X_tr, self.y_tr_cls,
            self.X_val, self.y_val_cls,
            self.X_te, self.y_te_cls,
            seed=0
        )
        best_c = results["Classical RBF-SVM (9 Feats)"]["test_metrics"]["best_c"]
        self.assertIn(best_c, [0.5, 1.0, 2.0])

    def test_svr_hyperparameter_selection(self):
        """Verifies RBF-SVR selects C from candidate grid {5.0, 10.0, 20.0}."""
        results = run_classical_regression_controls(
            self.X_tr, self.y_tr_reg,
            self.X_val, self.y_val_reg,
            self.X_te, self.y_te_reg,
            self.df_te,
            seed=0
        )
        best_c = results["Classical RBF-SVR (9 Feats)"]["test_metrics"]["best_c"]
        self.assertIn(best_c, [5.0, 10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
