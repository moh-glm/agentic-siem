
import unittest

from src.agents.agent1.evaluation import (
    evaluate_model,
    measure_prediction_performance,
)


class FakeModel:

    def predict(self, X):
        return [0, 0, 1, 1, 0]


class FakeProbabilityModel:

    classes_ = [0, 1]

    def predict(self, X):
        return [0, 0, 1, 1, 0]

    def predict_proba(self, X):
        return [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.10, 0.90],
            [0.60, 0.40],
        ]


class Agent1EvaluationTests(unittest.TestCase):

    def test_evaluation_metrics(self):
        y_test = [0, 0, 1, 1, 1]
        X_test = [1, 2, 3, 4, 5]

        results = evaluate_model(FakeModel(), X_test, y_test)

        self.assertAlmostEqual(results["accuracy"], 0.8)
        self.assertAlmostEqual(results["precision"], 1.0)
        self.assertAlmostEqual(results["recall"], 2 / 3)
        self.assertAlmostEqual(results["f1_score"], 0.8)

    def test_confusion_matrix(self):
        results = evaluate_model(
            FakeModel(),
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
        )

        self.assertEqual(
            results["confusion_matrix"],
            [[2, 0], [1, 2]],
        )

    def test_cybersecurity_metrics(self):
        results = evaluate_model(
            FakeModel(),
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
        )

        self.assertAlmostEqual(results["specificity"], 1.0)
        self.assertAlmostEqual(results["false_positive_rate"], 0.0)
        self.assertAlmostEqual(results["false_negative_rate"], 1 / 3)
        self.assertAlmostEqual(results["balanced_accuracy"], 5 / 6)

    def test_probability_metrics(self):
        results = evaluate_model(
            FakeProbabilityModel(),
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
        )

        self.assertIn("roc_auc", results)
        self.assertIn("average_precision", results)
        self.assertGreaterEqual(results["roc_auc"], 0.0)
        self.assertLessEqual(results["roc_auc"], 1.0)

    def test_model_without_probabilities(self):
        results = evaluate_model(
            FakeModel(),
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
        )

        self.assertNotIn("roc_auc", results)
        self.assertNotIn("average_precision", results)

    def test_single_class_dataset(self):
        results = evaluate_model(
            FakeProbabilityModel(),
            [1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0],
        )

        self.assertIsNone(results["roc_auc"])
        self.assertIsNone(results["average_precision"])

    def test_operational_error_metrics(self):
        results = evaluate_model(
            FakeModel(),
            [1, 2, 3, 4, 5],
            [0, 0, 1, 1, 1],
        )

        self.assertEqual(results["normal_flows"], 2)
        self.assertEqual(results["attack_flows"], 3)

        self.assertAlmostEqual(
            results["false_positives_per_1000_normal_flows"],
            0.0,
        )

        self.assertAlmostEqual(
            results["false_negatives_per_1000_attack_flows"],
            1000 / 3,
        )

    def test_prediction_performance(self):
        results = measure_prediction_performance(
            FakeModel(),
            [1, 2, 3, 4, 5],
        )

        self.assertEqual(results["records"], 5)
        self.assertGreaterEqual(results["prediction_time_seconds"], 0)
        self.assertGreater(results["throughput_flows_per_second"], 0)

    def test_invalid_prediction_length(self):
        class InvalidModel:

            def predict(self, X):
                return [0, 1]

        with self.assertRaises(ValueError):
            evaluate_model(
                InvalidModel(),
                [1, 2, 3],
                [0, 1, 0],
            )

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            evaluate_model(
                FakeModel(),
                [1, 2, 3],
                [0, 1, 2],
            )


if __name__ == "__main__":
    unittest.main()
