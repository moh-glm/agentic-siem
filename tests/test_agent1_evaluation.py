import unittest

from src.agents.agent1.evaluation import evaluate_model


class FakeModel:

    def predict(self, X):
        return [0, 0, 1, 1, 0]


class Agent1EvaluationTests(unittest.TestCase):

    def test_evaluation_metrics(self):
        y_test = [0, 0, 1, 1, 1]
        X_test = [1, 2, 3, 4, 5]

        results = evaluate_model(
            FakeModel(),
            X_test,
            y_test,
        )

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


if __name__ == "__main__":
    unittest.main()
