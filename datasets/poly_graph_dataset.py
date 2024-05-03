import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PolyGraphDataset(Dataset):
    """
    Graph from polyhedra of crystal
    """
    def __init__(self, poly_path, features_type=3):
        """
        features_type == 3 -> features: elements, vertex, types
        features_type == 2 -> features: elements, types
        """
        super().__init__()
        self.features_type = features_type
        # now is 3 features
        self.poly = pd.read_csv(poly_path)
        self.seebeck = pd.read_json(
            "/root/projects/ml-selection/data/raw_data/median_seebeck.json", orient='split'
        ).rename(columns={"Phase": "phase_id"})
        self.data = pd.merge(self.seebeck, self.poly, on="phase_id", how="inner").values.tolist()

    def __len__(self):
        """Return num of samples"""
        return len(self.data)

    def __getitem__(self, idx: int) -> list:
        """Return 1 graph and 1 seebeck"""
        seebeck = self.data[idx][2]
        elements = self.data[idx][3]

        if self.features_type == 3:
            vertex = self.data[idx][4]
            types = self.data[idx][5]
            graph = self.build_graph([elements, vertex, types])

        if self.features_type == 2:
            types = self.data[idx][4]
            graph = self.build_graph([elements, types])

        return [graph, seebeck]

    def build_graph(self, poly: list) -> Data:
        """
        Makes graph.
        Every node represent atom from polyhedra, witch has z-period and type of polyhedra.
        Graph is fully connected
        """
        if len(poly) == 3:
            poly_el, poly_vertex, poly_type = poly

            # create list with features for every node
            x_vector = []

            for i, d in enumerate(eval(poly_el)):
                x_vector.append([])
                x_vector[i].append(d)
                x_vector[i].append(eval(poly_type)[i])
                x_vector[i].append(eval(poly_vertex)[i])


            node_features = torch.tensor(x_vector)

        elif len(poly) == 2:
            poly_el, poly_type = poly

            # create list with features for every node
            x_vector = []

            for i, d in enumerate(eval(poly_el)):
                x_vector.append([])
                x_vector[i].append(d)
                x_vector[i].append(eval(poly_type)[0])

            node_features = torch.tensor(x_vector)

        edge_index = []

        for i in range(len(eval(poly_el))):
            for j in range(i + 1, len(eval(poly_el))):
                # graph is undirected, so we duplicate edge
                if i != j:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        graph_data = Data(
            x=node_features, edge_index=torch.tensor(edge_index).t().contiguous()
        )
        return graph_data