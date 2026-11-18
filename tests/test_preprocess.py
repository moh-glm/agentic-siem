import unittest

import pandas as pd

from src.data.preprocess import (
    validate_target,
    clean_port_columns,
    clean_attack_category,
    convert_port_column,
)


class PreprocessingTests(unittest.TestCase):

    def test_valid_target(self):
        df = pd.DataFrame({"Label": [0, 1, 0, 1]})
        result = validate_target(df)

        self.assertEqual(result["Label"].dtype, "int8")

    def test_missing_target_is_rejected(self):
        df = pd.DataFrame({"other": [1]})

        with self.assertRaises(ValueError):
            validate_target(df)

    def test_invalid_target_is_rejected(self):
        df = pd.DataFrame({"Label": [0, 1, 2]})

        with self.assertRaises(ValueError):
            validate_target(df)

    def test_missing_port_column_is_rejected(self):
        df = pd.DataFrame({"sport": [80]})

        with self.assertRaises(ValueError):
            clean_port_columns(df)

    def test_missing_attack_category_is_rejected(self):
        df = pd.DataFrame({"Label": [0]})

        with self.assertRaises(ValueError):
            clean_attack_category(df)

    def test_hexadecimal_ports_are_converted(self):
        values = pd.Series(["80", "0xc0a8", "0x000b", "-"])

        result = convert_port_column(values)

        self.assertEqual(result.iloc[0], 80)
        self.assertEqual(result.iloc[1], 49320)
        self.assertEqual(result.iloc[2], 11)
        self.assertTrue(pd.isna(result.iloc[3]))


if __name__ == "__main__":
    unittest.main()
