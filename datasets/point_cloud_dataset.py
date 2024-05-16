import polars as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from data.mendeleev_table import get_periodic_number


class PointCloudDataset(Dataset):
    """Dataset for PointNetwork"""
    def __init__(self, features: int):
        super().__init__()
        self.struct = pl.read_json('/root/projects/ml-selection/data/raw_data/rep_structures.json')
        self.seeb = pl.read_json('/root/projects/ml-selection/data/raw_data/median_seebeck.json')
        data = self.struct.join(self.seeb, on="phase_id", how="inner")
        self.data = [list(data.row(i)) for i in range(len(data))]
        self.features = features

    def __len__(self):
        """Return num of samples"""
        return len(self.data)

    def __getitem__(self, idx: int) -> list:
        """Return sample by idx """
        coordinates = self.data[idx][3]
        elements = self.data[idx][4]
        seebeck = self.data[idx][8]

        points_cloud = self.create_cloud(coordinates.copy(), elements.copy())
        return [points_cloud, seebeck]

    def create_cloud(self, coordinates: list, elements: list) -> Data:
        """
        Create set of points for each crystal structure
        """
        cloud = []
        els = [get_periodic_number(atom) for atom in elements]

        for idx, xyz in enumerate(coordinates):
            temp = xyz.copy()
            if self.features == 4:
                temp.append(els[idx])
            cloud.append(temp)

        edge_index = []

        for i in range(len(cloud)):
            for j in range(i + 1, len(cloud)):
                edge_index.append([i, j])

            if len(cloud) - (i + 1) == 0:
                edge_index.append([i, i])

        graph = Data(
            pos=torch.tensor(cloud),
            edge_index=torch.tensor(edge_index).t().contiguous(),
        )
        return graph
