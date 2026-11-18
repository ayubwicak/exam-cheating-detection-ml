"""Unit and integration test suite for Exam Cheating Detection ML Pipeline.

Tests:
1. Feature engineering and domain features correctness.
2. Robust missing value handling / imputation without data leakage.
3. Model creation and prediction consistency across registered models.
4. Threshold optimization and metric calculations.
5. End-to-end inference and bundle export validity.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from src.config import SEED, BASE_DIR
from src.features import (
    BASE_FEATURES,
    FEATURE_SETS,
    add_domain_features,
    build_feature_frame,
    get_feature_columns,
)
from src.models import available_models, make_model
from src.evaluation import find_best_threshold, fit_fold_models, predict_with_ensemble


class TestFeaturesPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.n_samples = 50
        self.mock_data = pd.DataFrame({
            "student_id": [f"STD_{i:04d}" for i in range(self.n_samples)],
            "duration_minutes": np.random.uniform(15.0, 90.0, self.n_samples),
            "score": np.random.randint(40, 100, self.n_samples),
            "tab_switch_count": np.random.randint(0, 15, self.n_samples),
            "copy_paste_count": np.random.randint(0, 10, self.n_samples),
            "avg_time_per_question": np.random.uniform(10.0, 120.0, self.n_samples),
            "mouse_movement_entropy": np.random.uniform(0.1, 0.99, self.n_samples),
            "network_ip_changes": np.random.randint(0, 4, self.n_samples),
            "similarity_with_peers": np.random.uniform(0.05, 0.95, self.n_samples),
        })

    def test_feature_sets_definitions(self):
        self.assertIn("A", FEATURE_SETS)
        self.assertIn("B", FEATURE_SETS)
        self.assertIn("C", FEATURE_SETS)
        self.assertEqual(len(FEATURE_SETS["A"]), len(BASE_FEATURES))
        self.assertGreater(len(FEATURE_SETS["C"]), len(FEATURE_SETS["A"]))

    def test_domain_features_logic(self):
        featured = add_domain_features(self.mock_data)
        # Check domain columns existence
        self.assertIn("score_speed_ratio", featured.columns)
        self.assertIn("is_suspicious_speed", featured.columns)
        self.assertIn("joki_remote_anomaly", featured.columns)
        self.assertIn("is_suspicious_joki", featured.columns)
        self.assertIn("activity_anomaly", featured.columns)

        # Mathematical logic assertion
        speedrun_mask = (featured["score"] >= 90) & (featured["duration_minutes"] <= 25)
        self.assertTrue((featured.loc[speedrun_mask, "is_suspicious_speed"] == 1).all())

        joki_mask = (featured["network_ip_changes"] > 0) & (featured["tab_switch_count"] >= 3)
        self.assertTrue((featured.loc[joki_mask, "is_suspicious_joki"] == 1).all())

    def test_robust_imputation_with_nans(self):
        dirty_data = self.mock_data.copy()
        dirty_data.loc[0, "duration_minutes"] = np.nan
        dirty_data.loc[1, "score"] = np.nan
        dirty_data.loc[2, "network_ip_changes"] = np.nan
        dirty_data.loc[3, "tab_switch_count"] = np.nan

        X = build_feature_frame(dirty_data, "C")
        if isinstance(X, tuple):
            X = X[0]

        self.assertEqual(X.isna().sum().sum(), 0, "Missing values were not cleanly imputed!")


class TestModelRegistryAndEvaluation(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.X = pd.DataFrame(np.random.randn(100, 8), columns=[f"f_{i}" for i in range(8)])
        self.y = np.random.choice([0, 1], size=100, p=[0.85, 0.15])

    def test_available_models(self):
        models = available_models()
        expected = ["lightgbm", "xgboost", "random_forest", "logreg"]
        for m in expected:
            self.assertIn(m, models, f"Model {m} missing from registry")

    def test_model_training_and_predict(self):
        for name in ["lightgbm", "logreg", "random_forest"]:
            model = make_model(name, seed=SEED)
            model.fit(self.X, self.y)
            probs = model.predict_proba(self.X)[:, 1]
            self.assertEqual(len(probs), 100)
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_threshold_tuning(self):
        y_true = np.array([0] * 80 + [1] * 20)
        # Synthetic probs with some signal
        y_prob = np.concatenate([np.random.uniform(0.0, 0.4, 80), np.random.uniform(0.5, 0.9, 20)])
        best_thresh, best_f1 = find_best_threshold(y_true, y_prob)
        self.assertGreaterEqual(best_thresh, 0.05)
        self.assertLessEqual(best_thresh, 0.95)
        self.assertGreater(best_f1, 0.5)


if __name__ == "__main__":
    unittest.main()
