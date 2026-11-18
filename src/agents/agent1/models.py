"""
Machine learning models for Agent 1.

Includes Gaussian Naive Bayes and Logistic Regression.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from src.agents.agent1.preprocessing import build_preprocessor


def build_naive_bayes_model():
    """
    Build a Gaussian Naive Bayes classification pipeline.
    """

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(
    scale_numerical=False,
    dense_output=True,
),
        ),
        (
            "classifier",
            GaussianNB(),
        ),
    ])


def build_logistic_regression_model():
    """
    Build a Logistic Regression classification pipeline.
    """

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(scale_numerical=True),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])
