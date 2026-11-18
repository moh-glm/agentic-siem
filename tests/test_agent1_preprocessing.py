import unittest

import numpy as np
import pandas as pd

from src.agents.agent1.preprocessing import (
    build_preprocessor,
    get_numerical_columns,
)
from src.data.preprocess import (
    CATEGORICAL_COLUMNS,
    get_feature_columns,
)


class Agent1PreprocessingTests(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "sport": [80.0, 443.0, np.nan, 53.0],
            "dsport": [8080.0, 443.0, 53.0, np.nan],
            "proto": ["tcp", "udp", "tcp", "icmp"],
            "state": ["CON", "FIN", "INT", "CON"],
            "dur": [0.1, 0.2, np.nan, 0.4],
            "service": ["http", "dns", "-", "http"],
        })

        for column in get_feature_columns():
            if column not in self.df.columns:
                self.df[column] = 0

        self.df = self.df[get_feature_columns()]

    def test_feature_groups(self):
        numerical = get_numerical_columns()

        self.assertEqual(len(numerical), 40)
        self.assertEqual(len(CATEGORICAL_COLUMNS), 3)

    def test_preprocessing_removes_missing_values(self):
        preprocessor = build_preprocessor()

        result = preprocessor.fit_transform(self.df)

        self.assertEqual(result.shape[0], 4)
        self.assertTrue(np.isfinite(result).all())

    def test_unknown_categories_are_supported(self):
        preprocessor = build_preprocessor()

        preprocessor.fit(self.df)

        new_data = self.df.copy()
        new_data.loc[0, "proto"] = "new_protocol"

        result = preprocessor.transform(new_data)

        self.assertEqual(result.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
