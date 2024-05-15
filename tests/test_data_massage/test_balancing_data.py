"""
Test cases to test oversampling (SMOTER) and undersampling
"""
import unittest

import polars as pd
import yaml

from data_massage.balancing_data.oversampling import make_oversampling
from data_massage.balancing_data.undersampling import make_undersampling


class TestBalancing(unittest.TestCase):
    def setUp(self):
        with open("/configs/config.yaml", "r") as yamlfile:
            yaml_file = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.str_path = yaml_file["structure_path"]
            self.seebeck_path = yaml_file["seebeck_path"]

        self.columns_str = ["atom", "distance"]
        self.columns_seeb = ["Seebeck coefficient"]

    def test_oversampling(self):
        df_str, df_seeb = make_oversampling(self.str_path, self.str_path)
        self.assertEqual(
            len(df_seeb),
            len(pd.read_csv(self.str_path)),
            "different size between not balanced and balanced data",
        )
        self.assertEqual(
            len(df_str),
            len(df_seeb),
            "different size between Seebeck and structure data",
        )
        self.assertEqual(pd.read_csv(self.str_path), df_str, "data not balanced")

    def test_undersampling(self):
        df_str, df_seeb = make_undersampling(2, self.str_path, self.seebeck_path)
        self.assertEqual(
            df_str.columns.tolist(),
            self.columns_str,
            "wrong columns name in structure DataFrame",
        )
        self.assertEqual(
            df_seeb.columns.tolist(),
            self.columns_seeb,
            "wrong columns name in Seebeck DataFrame",
        )
        self.assertGreater(
            len(pd.read_csv(self.str_path)),
            len(df_seeb),
            "number of samples in undersampling more then number in raw data",
        )


if __name__ == "__main__":
    unittest.main()