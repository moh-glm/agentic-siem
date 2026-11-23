
"""
Runtime interface for Agent 1.

Agent 1 classifies individual network flows using
the best model selected during validation.
"""

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.data.preprocess import (
    clean_port_columns,
    clean_categorical_columns,
    get_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODELS_DIR = (
    PROJECT_ROOT / "reports" / "results" / "agent1" / "models"
)

MODEL_FILE = "best_model.pkl"

EXPECTED_TARGET_MAPPING = {
    "0": "normal",
    "1": "attack",
}

SUPPORTED_MODELS = {
    "Gaussian Naive Bayes",
    "Logistic Regression",
}


class Agent1(BaseAgent):
    """Fast binary classification agent for network flows."""

    def __init__(
        self,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
    ):
        super().__init__("agent_1")

        self.models_dir = Path(models_dir)
        metadata_path = self.models_dir / "metadata.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Agent 1 metadata not found: {metadata_path}. "
                "Run the Agent 1 experiment first."
            )

        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                self.metadata = json.load(file)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                f"Invalid Agent 1 metadata file: {error}"
            ) from error

        if not isinstance(self.metadata, dict):
            raise ValueError("Agent 1 metadata must be a JSON object.")

        self.feature_columns = self.metadata.get("feature_columns")
        self.best_model_name = self.metadata.get("best_model")
        self.target_mapping = self.metadata.get("target_mapping")

        if (
            not isinstance(self.feature_columns, list)
            or not self.feature_columns
            or not all(
                isinstance(column, str) and column
                for column in self.feature_columns
            )
            or len(self.feature_columns) != len(set(self.feature_columns))
        ):
            raise ValueError(
                "Metadata must contain a valid, unique feature_columns list."
            )

        if self.feature_columns != get_feature_columns():
            raise ValueError(
                "Model feature columns do not match the current "
                "dataset feature definition."
            )

        if self.target_mapping != EXPECTED_TARGET_MAPPING:
            raise ValueError(
                "Invalid target mapping. Expected normal=0 "
                "and attack=1."
            )

        if self.best_model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported best model: {self.best_model_name}"
            )

        model_path = self.models_dir / MODEL_FILE

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}. "
                "Run the Agent 1 experiment first."
            )

        try:
            self.model = joblib.load(model_path)
        except Exception as error:
            raise ValueError(
                f"Could not load Agent 1 model: {error}"
            ) from error

        if not all(
            callable(getattr(self.model, attribute, None))
            for attribute in ("predict", "predict_proba")
        ):
            raise TypeError(
                "Loaded model must support predict() and predict_proba()."
            )

        if not hasattr(self.model, "classes_"):
            raise TypeError(
                "Loaded model does not expose fitted classes_."
            )

        if set(self.model.classes_) != {0, 1}:
            raise ValueError(
                "Loaded model must support both classes 0 and 1."
            )

        if not hasattr(self.model, "named_steps"):
            raise TypeError(
                "Loaded model must be a scikit-learn Pipeline."
            )

        classifier = self.model.named_steps.get("classifier")

        if classifier is None:
            raise TypeError(
                "Loaded model Pipeline has no classifier step."
            )

        expected_classifiers = {
            "Gaussian Naive Bayes": "GaussianNB",
            "Logistic Regression": "LogisticRegression",
        }

        expected_classifier = expected_classifiers[
            self.best_model_name
        ]

        if classifier.__class__.__name__ != expected_classifier:
            raise TypeError(
                f"Metadata expects {expected_classifier}, "
                f"but loaded model contains "
                f"{classifier.__class__.__name__}."
            )

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if (
            model_feature_count is not None
            and model_feature_count != len(self.feature_columns)
        ):
            raise ValueError(
                "Loaded model feature count does not match metadata."
            )

    @staticmethod
    def _error_response(
        input_id: str,
        message: str,
    ) -> dict:
        """Build a standardized error response."""

        return {
            "agent": "agent_1",
            "status": "error",
            "input_id": input_id,
            "prediction": None,
            "confidence": None,
            "evidence": {
                "error": message,
            },
        }

    def _prepare_input(
        self,
        data: Any,
    ) -> tuple[str, pd.DataFrame]:
        """Validate and preprocess one network flow."""

        if isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise ValueError(
                    "Agent 1 expects exactly one network flow."
                )

            if data.columns.duplicated().any():
                raise ValueError(
                    "Input contains duplicate feature columns."
                )

            input_id = str(data.index[0])
            flow = data.copy()

        elif isinstance(data, dict):
            input_id = str(data.get("input_id", "unknown"))

            flow = pd.DataFrame([
                {
                    column: data[column]
                    for column in self.feature_columns
                    if column in data
                }
            ])

        else:
            raise TypeError(
                "Input must be a dictionary or a pandas DataFrame."
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in flow.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required features: {missing_columns}"
            )

        flow = flow[self.feature_columns].copy()

        flow = clean_port_columns(flow)
        flow = clean_categorical_columns(flow)

        return input_id, flow

    @staticmethod
    def _classify(
        model: Any,
        flow: pd.DataFrame,
    ) -> tuple[int, float]:
        """Return a validated prediction and its class probability."""

        predictions = model.predict(flow)

        if len(predictions) != 1:
            raise ValueError(
                "Model must return exactly one prediction."
            )

        prediction = int(predictions[0])

        if prediction not in (0, 1):
            raise ValueError(
                f"Invalid predicted class: {prediction}"
            )

        classes = list(model.classes_)

        if set(classes) != {0, 1}:
            raise ValueError(
                "Model classes must contain exactly 0 and 1."
            )

        probabilities = model.predict_proba(flow)

        if len(probabilities) != 1:
            raise ValueError(
                "Model must return probabilities for exactly one flow."
            )

        probabilities = probabilities[0]

        if len(probabilities) != len(classes):
            raise ValueError(
                "Probability count does not match model classes."
            )

        probabilities = [
            float(probability)
            for probability in probabilities
        ]

        if not all(
            math.isfinite(probability)
            and 0.0 <= probability <= 1.0
            for probability in probabilities
        ):
            raise ValueError(
                "Model returned invalid probabilities."
            )

        if not math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError(
                "Model probabilities must sum to 1."
            )

        confidence = probabilities[classes.index(prediction)]

        return prediction, confidence

    def predict(self, data: Any) -> dict:
        """Classify one network flow and return a standard response."""

        input_id = "unknown"

        if isinstance(data, dict):
            input_id = str(data.get("input_id", "unknown"))

        elif isinstance(data, pd.DataFrame) and len(data) == 1:
            input_id = str(data.index[0])

        try:
            input_id, flow = self._prepare_input(data)

            best_prediction, confidence = self._classify(
                self.model,
                flow,
            )

            selected_label = self.target_mapping[
                str(best_prediction)
            ]

            prediction = {
                "label": selected_label,
                "naive_bayes": (
                    selected_label
                    if self.best_model_name == "Gaussian Naive Bayes"
                    else None
                ),
                "logistic_regression": (
                    selected_label
                    if self.best_model_name == "Logistic Regression"
                    else None
                ),
            }

            evidence = {
                "agreement": None,
                "selected_model": self.best_model_name,
            }

            return self.build_response(
                input_id=input_id,
                prediction=prediction,
                confidence=confidence,
                evidence=evidence,
            )

        except Exception as error:
            return self._error_response(
                input_id,
                str(error),
            )
