import polars as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PolyGraphDataset(Dataset):
    """
    Graph from polyhedra of crystal
    """
    def __init__(self, poly_path, features_type: int = 3, add_temperature: bool = False):
        """
        Parameters
        ----------
        poly_path: str
            Path to poly data
        features_type: int, optional
            features_type == 2 -> features: elements, types
            features_type == 3 -> features: elements, vertex, types
        add_temperature : bool, optional
            Add temperature to features
        """
        super().__init__()
        self.features_type = features_type
        # now is 3 features
        self.poly = pd.read_csv(poly_path)
        self.temperature = add_temperature
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

        if self.features_type == 2:
            types = self.data[idx][4]
            if self.temperature:
                t = self.data[idx][5]
                graph = self.build_graph([elements, types, t])
            else:
                graph = self.build_graph([elements, types])

        if self.features_type == 3:
            vertex = self.data[idx][4]
            types = self.data[idx][5]
            if self.temperature:
                types = self.data[idx][4]
                t = self.data[idx][5]
                graph = self.build_graph([elements, types, t])
            else:
                graph = self.build_graph([elements, vertex, types])

        if self.features_type == 4:
            vertex = self.data[idx][4]
            types = self.data[idx][5]
            t = self.data[idx][6]
            graph = self.build_graph([elements, vertex, types, t])

        return [graph, seebeck]

    def build_graph(self, poly: list) -> Data:
        """
        Makes graph.
        Every node represent atom from polyhedra, witch has z-period and type of polyhedra.
        Graph is fully connected
        """
        if len(poly) == 2:
            poly_el, poly_type = poly
        elif len(poly) == 3:
            if self.temperature:
                poly_el, poly_type, t = poly
            else:
                poly_el, poly_vertex, poly_type = poly
        elif len(poly) == 4:
            poly_el, poly_vertex, poly_type, t = poly

        # create list with features for every node
        x_vector = []

        for i, d in enumerate(eval(poly_el)):
            x_vector.append([])
            x_vector[i].append(d)
            # if descriptor is vector of counts
            if len(eval(poly_el)) > len(eval(poly_type)):
                x_vector[i].append(eval(poly_type)[0])
            else:
                x_vector[i].append(eval(poly_type)[i])
            if len(poly) == 3:
                if self.temperature:
                    x_vector[i].append(t)
                else:
                    x_vector[i].append(eval(poly_vertex)[i])
            elif len(poly) == 4:
                x_vector[i].append(eval(poly_vertex)[i])
                x_vector[i].append(t)

        node_features = torch.tensor(x_vector)

        edge_index = []

        for i in range(len(eval(poly_el))):
            for j in range(i + 1, len(eval(poly_el))):
                # graph is undirected, so we duplicate edge
                if i != j:
                    edge_index.append([i, j])
                    # edge_index.append([j, i])

        graph_data = Data(
            x=node_features, edge_index=torch.tensor(edge_index).t().contiguous()
        )
        return graph_data