"""
Test for build graph by CrystalGraphVectorsDataset
"""

import unittest

import polars as pd
from torch_geometric.data import Data

from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset


class TestRequestMPDS(unittest.TestCase):
    def setUp(self):
        self.total = pd.read_csv(
            "/root/projects/ml-selection/data/processed_data/under_str.csv",
        )
        self.seebeck = pd.read_csv(
            "/root/projects/ml-selection/data/processed_data/under_seeb.csv",
        )
        self.data = pd.concat(
            [self.seebeck["Seebeck coefficient"], self.total], axis=1
        ).values.tolist()
        self.dataset = CrystalGraphVectorsDataset()

    def test_build_graph(self):
        graph = self.dataset.build_graph(self.total.values.tolist()[0])

        self.assertEqual(type(graph), Data, "incorrect type of graph")
        self.assertEqual(
            len(graph.x) * len(graph.x) - len(graph.x),
            len(graph.edge_index[0]),
            "incorrectly defined edges between atoms",
        )
        self.assertEqual(len(graph.x[0]), 2, "incorrect number of features for nodes")
