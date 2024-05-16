"""
Test for build graph by CrystalGraphVectorsDataset
"""

import unittest

import polars as pl
from torch_geometric.data import Data

from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset


class TestRequestMPDS(unittest.TestCase):
    def setUp(self):
        self.total = pl.read_csv(
            "/root/projects/ml-selection/data/processed_data/rep_vect_str.csv",
        )
        self.seebeck = pl.read_csv(
            "/root/projects/ml-selection/data/processed_data/rep_vect_seebeck.csv",
        )

        self.dataset = CrystalGraphVectorsDataset()

    def test_build_graph(self):
        graph = self.dataset.build_graph(self.total.row(0))

        self.assertEqual(type(graph), Data, "incorrect type of graph")
        self.assertEqual(
            len(graph.x) * len(graph.x) - len(graph.x),
            len(graph.edge_index[0]),
            "incorrectly defined edges between atoms",
        )
        self.assertEqual(len(graph.x[0]), 2, "incorrect number of features for nodes")
