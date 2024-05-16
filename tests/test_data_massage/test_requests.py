"""
Test cases to test request to MPDS database
"""
import unittest

import polars as pl
import yaml

from data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS


class TestRequestMPDS(unittest.TestCase):
    def setUp(self):
        with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
            self.key = yaml.load(yamlfile, Loader=yaml.FullLoader)["api_key"]
        self.client_handler = RequestMPDS(dtype=1, api_key=self.key)
        self.seebeck_columns = ["Phase", "Formula", "Seebeck coefficient"]
        self.structure_columns = [
            'phase_id',
            'occs_noneq',
            'cell_abc',
            'sg_n',
            'basis_noneq',
            'els_noneq',
            'entry',
            'temperature'
        ]

    def test_make_request(self):
        seebeck = self.client_handler.make_request(is_seebeck=True)
        self.assertEqual(type(seebeck), type(pl.DataFrame()), "incorrect type")
        self.assertEqual(
            seebeck.columns,
            self.seebeck_columns,
            "incorrect columns in DataFrame",
        )
        self.assertEqual(
            len(seebeck), 8182, "wrong number of samples in answer"
        )

        structures = self.client_handler.make_request(
            is_structure=True, phases=list(seebeck["Phase"])
        )
        self.assertEqual(type(structures), type(pl.DataFrame()), "incorrect type")
        self.assertEqual(
            structures.columns,
            self.structure_columns,
            "incorrect columns in DataFrame",
        )
        self.assertNotEqual(len(structures), 0, "empty answer")

        self.assertGreater(
            len(set(seebeck["Phase"])),
            len(set(structures["phase_id"])),
            "set phases in structures more then set phases in Seebeck values",
        )
