import polars as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class CrystalGraphVectorsDataset(Dataset):
    """
    Dataset consist of structures represented in graph format. Every node represent atom,
    which have features: periodic number of atom and distance to origin.
    """

    def __init__(self):
        super().__init__()
        self.data = pl.read_json(
            "ml_selection/data/processed_data/rep_vect_str_clear.json",
        )
        self.data = [list(self.data.row(i)) for i in range(len(self.data))]

    def __len__(self):
        "Return num of samples"
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        "Return 1 graph and 1 seebeck"
        atoms = self.data[idx][1]
        distance = self.data[idx][2]
        seebeck = self.data[idx][0]

        graph = self.build_graph([atoms, distance])
        return graph, seebeck

    def build_graph(self, crystal_data: list) -> Data:
        """
        Makes graph.
        Every node represent atom, which have features:
        - periodic number of atom
        - distance to origin.
        Graph is fully connected.
        """
        atoms, distance = crystal_data

        # create list with features for every node
        x_vector = []

        for i, d in enumerate(eval(distance)):
            x_vector.append([])
            x_vector[i].append(eval(atoms)[i])
            x_vector[i].append(d)

        node_features = torch.tensor(x_vector)

        edge_index = []

        for i in range(len(eval(atoms))):
            for j in range(i + 1, len(eval(atoms))):
                # graph is undirected, so we duplicate edge
                if i != j:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        graph_data = Data(
            x=node_features, edge_index=torch.tensor(edge_index).t().contiguous()
        )
        return graph_data
