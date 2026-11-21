
"""
Runtime interface for Agent 1.

Agent 1 classifies individual network flows using
Gaussian Naive Bayes and Logistic Regression.
"""

import json
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

MODEL_FILES = {
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}

EXPECTED_TARGET_MAPPING = {
    "0": "normal",
    "1": "attack",
}


class Agent1(BaseAgent):
    """Fast binary classification agent for network flows."""

    def __init__(self, models_dir: Path | str = DEFAULT_MODELS_DIR):
        super().__init__("agent_1")

        self.models_dir = Path(models_dir)

        metadata_path = self.models_dir / "metadata.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Agent 1 metadata not found: {metadata_path}. "
                "Run the Agent 1 experiment first."
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.feature_columns = self.metadata.get("feature_columns")
        self.best_model_name = self.metadata.get("best_model")

        if self.feature_columns != get_feature_columns():
            raise ValueError(
                "Model feature columns do not match the current "
                "dataset feature definition."
            )

        self.target_mapping = self.metadata.get("target_mapping")

        if self.target_mapping != EXPECTED_TARGET_MAPPING:
            raise ValueError(
                "Invalid target mapping in Agent 1 metadata. "
                "Expected normal=0 and attack=1."
            )

        if self.best_model_name not in MODEL_FILES:
            raise ValueError(
                f"Unsupported best model: {self.best_model_name}"
            )

        self.models = {}

        for model_name, filename in MODEL_FILES.items():
            model_path = self.models_dir / filename

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Trained model not found: {model_path}"
                )

            self.models[model_name] = joblib.load(model_path)

    @staticmethod
    def _error_response(input_id: str, message: str) -> dict:
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

    def _prepare_input(self, data: Any) -> tuple[str, pd.DataFrame]:
        """Validate and preprocess one network flow."""

        if isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise ValueError(
                    "Agent 1 expects exactly one network flow."
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
    def _classify(model, flow: pd.DataFrame) -> tuple[int, float]:
        """Return predicted class and its probability."""

        prediction = int(model.predict(flow)[0])

        probabilities = model.predict_proba(flow)[0]

        classes = list(model.classes_)

        if prediction not in classes:
            raise ValueError(
                f"Predicted class {prediction} is not in model classes."
            )

        confidence = float(
            probabilities[classes.index(prediction)]
        )

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

            predictions = {}

            for model_name, model in self.models.items():
                predictions[model_name] = self._classify(
                    model,
                    flow,
                )

            nb_prediction = predictions["Gaussian Naive Bayes"][0]
            lr_prediction = predictions["Logistic Regression"][0]

            best_prediction, confidence = predictions[
                self.best_model_name
            ]

            prediction = {
                "label": self.target_mapping[str(best_prediction)],
                "naive_bayes": self.target_mapping[str(nb_prediction)],
                "logistic_regression": self.target_mapping[str(lr_prediction)],
            }

            evidence = {
                "agreement": nb_prediction == lr_prediction,
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
