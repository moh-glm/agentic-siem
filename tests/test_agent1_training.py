import unittest

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.agents.agent1.train import (
    split_dataset,
    train_model,
)

from src.agents.agent1.models import (
    build_logistic_regression_model,
)

from src.data.preprocess import get_feature_columns


class Agent1TrainingTests(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "Label": [0] * 900 + [1] * 100,
        })

        for column in get_feature_columns():
            if column in ["proto", "state", "service"]:
                self.df[column] = "unknown"
            else:
                self.df[column] = 0
    def test_split_sizes(self):
        X_train, X_val, X_test, y_train, y_val, y_test = (
            split_dataset(self.df)
        )

        self.assertEqual(len(X_train), 700)
        self.assertEqual(len(X_val), 150)
        self.assertEqual(len(X_test), 150)

    def test_class_distribution_is_preserved(self):
        _, _, _, y_train, y_val, y_test = split_dataset(self.df)

        for target in [y_train, y_val, y_test]:
            self.assertAlmostEqual(target.mean(), 0.1, places=2)

    def test_invalid_split_is_rejected(self):
        with self.assertRaises(ValueError):
            split_dataset(self.df, test_size=0.6, validation_size=0.5)

    def test_train_model_returns_fitted_model(self):
        X_train, X_val, _, y_train, y_val, _ = split_dataset(self.df)

        model = build_logistic_regression_model()

        trained_model, results = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        self.assertIsInstance(trained_model, Pipeline)
        self.assertTrue(
            hasattr(
                trained_model.named_steps["classifier"],
                "coef_",
            )
        )
        self.assertIn("accuracy", results)

    def test_validation_metrics_are_valid(self):
        X_train, X_val, _, y_train, y_val, _ = split_dataset(self.df)

        model = build_logistic_regression_model()

        _, results = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)

        self.assertEqual(
            len(results["confusion_matrix"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
