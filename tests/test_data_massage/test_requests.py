"""
Test cases to test request to MPDS database
"""
import unittest

import pandas as pd
import yaml

from data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS


class TestRequestMPDS(unittest.TestCase):
    def setUp(self):
        with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
            self.key = yaml.load(yamlfile, Loader=yaml.FullLoader)["api_key"]
        self.client_handler = RequestMPDS(dtype=1, api_key=self.key)
        self.seebeck_columns = ["Phase", "Formula", "Seebeck coefficient"]
        self.structure_columns = [
            "phase_id",
            "occs_noneq",
            "cell_abc",
            "sg_n",
            "basis_noneq",
            "els_noneq",
        ]

    def test_make_request(self):
        seebeck = self.client_handler.make_request(is_seebeck=True)
        self.assertEqual(type(seebeck), type(pd.DataFrame()), "incorrect type")
        self.assertEqual(
            seebeck.columns.tolist(),
            self.seebeck_columns,
            "incorrect columns in DataFrame",
        )
        self.assertEqual(
            len(seebeck.values.tolist()), 8182, "wrong number of samples in answer"
        )

        structures = self.client_handler.make_request(
            is_structure=True, phases=seebeck["Phase"].values.tolist()
        )
        self.assertEqual(type(structures), type(pd.DataFrame()), "incorrect type")
        self.assertEqual(
            structures.columns.tolist(),
            self.structure_columns,
            "incorrect columns in DataFrame",
        )
        self.assertNotEqual(len(structures.values.tolist()), 0, "empty answer")

        self.assertGreater(
            len(set(seebeck["Phase"].values.tolist())),
            len(set(structures["phase_id"].values.tolist())),
            "set phases in structures more then set phases in Seebeck values",
        )
