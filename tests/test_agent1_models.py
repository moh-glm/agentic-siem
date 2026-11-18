import unittest

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from src.agents.agent1.models import (
    build_naive_bayes_model,
    build_logistic_regression_model,
)


class Agent1ModelTests(unittest.TestCase):

    def test_naive_bayes_pipeline(self):
        model = build_naive_bayes_model()

        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(
            model.named_steps["classifier"],
            GaussianNB,
        )

    def test_logistic_regression_pipeline(self):
        model = build_logistic_regression_model()

        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(
            model.named_steps["classifier"],
            LogisticRegression,
        )

    def test_logistic_regression_balances_classes(self):
        model = build_logistic_regression_model()
        classifier = model.named_steps["classifier"]

        self.assertEqual(classifier.class_weight, "balanced")


if __name__ == "__main__":
    unittest.main()
