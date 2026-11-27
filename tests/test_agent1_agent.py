
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.agents.agent1.agent import Agent1, MODEL_FILE
from src.agents.agent1.models import build_logistic_regression_model
from src.data.preprocess import get_feature_columns


class Agent1RuntimeTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name)

        self.feature_columns = get_feature_columns()

        # Create a small synthetic training dataset.
        rows = []

        for index in range(20):
            row = {
                column: 0.0
                for column in self.feature_columns
            }

            row.update({
                "proto": "tcp" if index % 2 else "udp",
                "state": "fin" if index % 2 else "con",
                "service": "http" if index % 2 else "-",
                "sport": 1000 + index,
                "dsport": 80,
                "dur": float(index),
                "sbytes": float(index * 10),
                "dbytes": float(index * 5),
            })

            rows.append(row)

        X = pd.DataFrame(rows)
        y = pd.Series([0, 1] * 10)

        # Train and save the selected model.
        self.model = build_logistic_regression_model()
        self.model.fit(X, y)

        joblib.dump(
            self.model,
            self.models_dir / MODEL_FILE,
        )

        metadata = {
            "dataset": "UNSW-NB15",
            "best_model": "Logistic Regression",
            "model_file": MODEL_FILE,
            "selection_metric": "validation_f1",
            "validation_f1": 0.9,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "target_mapping": {
                "0": "normal",
                "1": "attack",
            },
        }

        self.metadata_path = self.models_dir / "metadata.json"

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file)

        self.agent = Agent1(models_dir=self.models_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_flow(self):
        flow = {
            column: 0.0
            for column in self.feature_columns
        }

        flow.update({
            "input_id": "flow_test_001",
            "proto": " TCP ",
            "state": " FIN ",
            "service": " HTTP ",
            "sport": "0x1F90",
            "dsport": "80",
        })

        return flow

    def test_valid_flow_returns_standard_response(self):
        result = self.agent.predict(self.make_flow())

        self.assertEqual(result["agent"], "agent_1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["input_id"], "flow_test_001")

        self.assertIn("label", result["prediction"])
        self.assertIn("naive_bayes", result["prediction"])
        self.assertIn(
            "logistic_regression",
            result["prediction"],
        )

        self.assertIn(
            result["prediction"]["label"],
            ["normal", "attack"],
        )

        self.assertIsNone(result["prediction"]["naive_bayes"])
        self.assertIsNotNone(
            result["prediction"]["logistic_regression"]
        )

        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        self.assertIsNone(result["evidence"]["agreement"])
        self.assertEqual(
            result["evidence"]["selected_model"],
            "Logistic Regression",
        )

    def test_missing_feature_returns_error(self):
        flow = self.make_flow()
        del flow["dur"]

        result = self.agent.predict(flow)

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["prediction"])
        self.assertIn(
            "Missing required features",
            result["evidence"]["error"],
        )

    def test_invalid_input_returns_error(self):
        result = self.agent.predict("invalid input")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["input_id"], "unknown")

    def test_dataframe_input_is_supported(self):
        flow = self.make_flow()
        input_id = flow.pop("input_id")

        dataframe = pd.DataFrame(
            [flow],
            index=[input_id],
        )

        result = self.agent.predict(dataframe)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["input_id"], input_id)

    def test_invalid_target_mapping_is_rejected(self):
        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        metadata["target_mapping"] = {
            "0": "attack",
            "1": "normal",
        }

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file)

        with self.assertRaisesRegex(
            ValueError,
            "Invalid target mapping",
        ):
            Agent1(models_dir=self.models_dir)

    def test_model_type_mismatch_is_rejected(self):
        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        # The saved model is Logistic Regression,
        # but metadata incorrectly declares Gaussian Naive Bayes.
        metadata["best_model"] = "Gaussian Naive Bayes"

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file)

        with self.assertRaisesRegex(
            TypeError,
            "Metadata expects GaussianNB",
        ):
            Agent1(models_dir=self.models_dir)

    def test_invalid_metadata_fields_are_rejected(self):
        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            original_metadata = json.load(file)

        invalid_cases = [
            ("dataset", "INVALID", "dataset metadata"),
            ("model_file", "wrong.pkl", "model_file metadata"),
            (
                "selection_metric",
                "accuracy",
                "selection_metric metadata",
            ),
            ("feature_count", 999, "feature_count"),
            ("validation_f1", 1.5, "validation_f1"),
            ("validation_f1", float("nan"), "validation_f1"),
        ]

        for field, invalid_value, expected_error in invalid_cases:
            with self.subTest(field=field, value=invalid_value):
                metadata = original_metadata.copy()
                metadata[field] = invalid_value

                with self.metadata_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(metadata, file)

                with self.assertRaisesRegex(
                    ValueError,
                    expected_error,
                ):
                    Agent1(models_dir=self.models_dir)

        # Restore valid metadata.
        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(original_metadata, file)

    def test_missing_metadata_fields_are_rejected(self):
        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            original_metadata = json.load(file)

        required_fields = [
            "dataset",
            "best_model",
            "model_file",
            "selection_metric",
            "validation_f1",
            "feature_count",
            "feature_columns",
            "target_mapping",
        ]

        for field in required_fields:
            with self.subTest(field=field):
                metadata = original_metadata.copy()
                del metadata[field]

                with self.metadata_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(metadata, file)

                with self.assertRaises(ValueError):
                    Agent1(models_dir=self.models_dir)

        # Restore valid metadata.
        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(original_metadata, file)


if __name__ == "__main__":
    unittest.main()
