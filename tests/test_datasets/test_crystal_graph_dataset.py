"""
Test for build graph by CrystalGraphDataset
"""

import unittest

import polars as pd
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datasets.crystal_graph_dataset import CrystalGraphDataset


class TestRequestMPDS(unittest.TestCase):
    def setUp(self):
        # DataFrame with columns:
        # ['phase_id', 'Formula', 'Seebeck coefficient', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']
        self.dfrm = self.data_csv = pd.read_csv(
            "/root/projects/ml-selection/data/processed_data/total.csv"
        )
        self.dataset = CrystalGraphDataset()

    def test_build_graph(self):
        graph, seebeck = self.dataset.build_graph(
            [list(self.dfrm.row(i)) for i in range(len(self.dfrm))]
        )
        self.assertEqual(type(seebeck), float, "incorrect type of Seebeck")
        self.assertEqual(type(graph), Data, "incorrect type of graph")
        self.assertEqual(
            len(graph.x) * len(graph.x) - len(graph.x),
            len(graph.edge_index[0]),
            "incorrectly defined edges between atoms",
        )
        self.assertEqual(len(graph.x[0]), 4, "incorrect number of features for nodes")
        self.assertEqual(
            len(graph.edge_index[0]),
            len(graph.edge_attr),
            "incorrect number of features for edges",
        )

    def test_get_item(self):
        dataloader = DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0
        )
        for batch in dataloader:
            self.assertEqual(type(batch[0].x), Tensor, "incorrect type")
            self.assertEqual(
                len(batch[0].x[0]), 4, "incorrect number of features for nodes"
            )
            break
