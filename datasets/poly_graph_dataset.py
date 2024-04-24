import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PolyGraphDataset(Dataset):
    """
    Graph from polyhedra of crystal
    """
    def __init__(self):
        super().__init__()
        self.poly = pd.read_csv(
            "/root/projects/ml-selection/data/processed_data/large_poly_descriptor.csv",
        )
        self.seebeck = pd.read_json(
            "/root/projects/ml-selection/data/raw_data/median_seebeck.json", orient='split'
        ).rename(columns={"Phase": "phase_id"})
        self.data = pd.merge(self.seebeck, self.poly, on="phase_id", how="inner").sample(frac=1).values.tolist()

    def __len__(self):
        "Return num of samples"
        return len(self.data)

    def __getitem__(self, idx: int) -> list:
        "Return 1 graph and 1 seebeck"
        seebeck = self.data[idx][2]
        elements = self.data[idx][3]
        types = self.data[idx][4]

        graph = self.build_graph([elements, types])
        return [graph, seebeck]

    def build_graph(self, poly: list) -> Data:
        """
        Makes graph.
        Every node represent atom from polyhedra, witch has z-period and type of polyhedra.
        Graph is fully connected
        """
        poly_el, poly_type = poly

        # create list with features for every node
        x_vector = []

        for i, d in enumerate(eval(poly_type)):
            x_vector.append([])
            x_vector[i].append(eval(poly_el)[i])
            x_vector[i].append(d)

        node_features = torch.tensor(x_vector)

        edge_index = []

        for i in range(len(eval(poly_type))):
            for j in range(i + 1, len(eval(poly_type))):
                # graph is undirected, so we duplicate edge
                if i != j:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        graph_data = Data(
            x=node_features, edge_index=torch.tensor(edge_index).t().contiguous()
        )
        return graph_data