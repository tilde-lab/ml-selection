"""
Test cases to test DataHandler class
"""
import unittest

import pandas as pd
import yaml

from data_massage.calculate_median_value import seebeck_median_value
from data_massage.data_handler import DataHandler


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        with open("/root/projects/ml-selection/config.yaml", "r") as yamlfile:
            self.api_key = yaml.load(yamlfile, Loader=yaml.FullLoader)["api_key"]

        self.raw_path = "/root/projects/ml-selection/data/raw_data/"
        self.processed_path = "/root/projects/ml-selection/data/processed_data/"
        self.handler = DataHandler(True, self.api_key)
        self.phases = [
            65536,
            24579,
            81924,
            147464,
            10,
            8203,
            12,
            13,
            8204,
            8207,
            24588,
            65548,
            65554,
            24597,
            8218,
            147483,
            147485,
            106527,
            131104,
            49186,
            8227,
            106532,
            106533,
            40,
            147501,
            106546,
            106547,
            49204,
            49205,
            131123,
        ]

    def test_just_seebeck(self):
        columns = ["Phase", "Formula", "Seebeck coefficient"]
        answer = self.handler.just_seebeck(200, -150)

        self.assertEqual(type(answer), type(pd.DataFrame()), "incorrect type")
        self.assertEqual(
            answer.columns.tolist(), columns, "incorrect columns in DataFrame"
        )
        self.assertNotEqual(len(answer.values.tolist()), 0, "empty answer")
        self.assertGreater(
            len(answer.values.tolist()), 7000, "suspiciously little data"
        )

    def test_to_order_disordered_str(self):
        columns = ["phase_id", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]
        answer = self.handler.to_order_disordered_str(self.phases, False)

        self.assertEqual(
            answer.columns.tolist(), columns, "incorrect columns in DataFrame"
        )
        self.assertEqual(len(answer), 423, "incorrect number of samples in DataFrame")

    def test_seebeck_median_value(self):
        seebeck = self.handler.just_seebeck(200, -150)
        answer = seebeck_median_value(seebeck, seebeck["Phase"].tolist())[
            "Seebeck coefficient"
        ].tolist()
        self.assertGreater(
            len(answer),
            len(set(answer)),
            "size of set of Seebeck median value >= size of list with repetitive Seebeck value",
        )

    def test_loop_of_collection(self):
        seebeck_dfrm = self.handler.just_seebeck(
            max_value=200, min_value=-150, is_uniq_phase_id=False
        )
        self.assertEqual(
            len(seebeck_dfrm.values.tolist()),
            7112,
            "expected another number of Seebeck values",
        )

        median_seebeck = seebeck_median_value(
            seebeck_dfrm, set(seebeck_dfrm["Phase"].tolist())
        )
        self.assertGreater(
            len(median_seebeck),
            len(set(median_seebeck)),
            "size of set of Seebeck median value >= size of list with repetitive Seebeck value",
        )

        structures_dfrm = self.handler.to_order_disordered_str(
            phases=set(seebeck_dfrm["Phase"].tolist()), is_uniq_phase_id=True
        )
        result_dfrm = self.handler.add_seebeck_by_phase_id(
            median_seebeck, structures_dfrm
        )
        self.assertEqual(
            len(result_dfrm.values.tolist()),
            len(structures_dfrm.values.tolist()),
            "different size of ordered structures after adding Seebeck value",
        )

        dfrm_str, dfrm_seeb = self.handler.to_cut_vectors_struct(dfrm=result_dfrm)
        self.assertEqual(
            len(dfrm_str.values.tolist()),
            2886,
            "wrong size of DataFrame after convert structures to vectors",
        )


if __name__ == "__main__":
    unittest.main()
