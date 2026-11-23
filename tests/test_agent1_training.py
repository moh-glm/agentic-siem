import unittest

import pandas as pd

from sklearn.pipeline import Pipeline

from src.agents.agent1.train import split_dataset, train_model
from src.agents.agent1.models import build_logistic_regression_model
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
                self.df[column] = range(1000)

        self.df = self.df.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)

    def test_split_sizes(self):
        X_train, X_val, X_test, y_train, y_val, y_test = (
            split_dataset(self.df)
        )

        self.assertAlmostEqual(len(X_train), 700, delta=35)
        self.assertAlmostEqual(len(X_val), 150, delta=15)
        self.assertAlmostEqual(len(X_test), 150, delta=15)

    def test_class_distribution_is_preserved(self):
        _, _, _, y_train, y_val, y_test = split_dataset(self.df)

        for target in [y_train, y_val, y_test]:
            self.assertAlmostEqual(target.mean(), 0.1, delta=0.04)


    def test_duplicate_features_never_cross_splits(self):
        duplicated_df = pd.concat(
            [
                self.df,
                self.df.iloc[[0, 1, 2]],
            ],
            ignore_index=True,
        )

        X_train, X_val, X_test, _, _, _ = split_dataset(
            duplicated_df
        )

        train_hashes = set(
            pd.util.hash_pandas_object(
                X_train, index=False
            )
        )
        val_hashes = set(
            pd.util.hash_pandas_object(
                X_val, index=False
            )
        )
        test_hashes = set(
            pd.util.hash_pandas_object(
                X_test, index=False
            )
        )

        self.assertEqual(len(train_hashes & val_hashes), 0)
        self.assertEqual(len(train_hashes & test_hashes), 0)
        self.assertEqual(len(val_hashes & test_hashes), 0)

    def test_conflicting_duplicate_labels_are_rejected(self):
        duplicate = self.df.iloc[[0]].copy()
        duplicate["Label"] = 1 - int(duplicate["Label"].iloc[0])

        conflicting_df = pd.concat(
            [self.df, duplicate],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "conflicting labels",
        ):
            split_dataset(conflicting_df)

    def test_invalid_split_is_rejected(self):
        with self.assertRaises(ValueError):
            split_dataset(self.df, test_size=0.6, validation_size=0.5)

    def test_train_model_returns_fitted_model(self):
        X_train, X_val, _, y_train, y_val, _ = split_dataset(self.df)

        model = build_logistic_regression_model()

        trained_model, results = train_model(
            model, X_train, y_train, X_val, y_val
        )

        self.assertIsInstance(trained_model, Pipeline)
        self.assertTrue(hasattr(
            trained_model.named_steps["classifier"], "coef_"
        ))
        self.assertIn("accuracy", results)

    def test_validation_metrics_are_valid(self):
        X_train, X_val, _, y_train, y_val, _ = split_dataset(self.df)

        model = build_logistic_regression_model()

        _, results = train_model(
            model, X_train, y_train, X_val, y_val
        )

        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            self.assertGreaterEqual(results[metric], 0.0)
            self.assertLessEqual(results[metric], 1.0)

        self.assertEqual(len(results["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
