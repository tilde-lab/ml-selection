import polars as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from data_massage.normalization.normalization import make_normalization
import yaml


CONF = "/root/projects/ml-selection/configs/config.yaml"


class PolyGraphDataset(Dataset):
    """
    Graph from polyhedra of crystal
    """

    def __init__(
        self, poly_path, features_type: int = 3, add_temperature: bool = False, just_mp: bool = False
    ):
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
        with open(CONF, "r") as yamlfile:
            yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
            path_mpds = yaml_f["raw_mpds"]
        self.features_type = features_type
        # now is 3 features
        poly = pl.read_json(poly_path)
        self.poly = poly.with_columns(pl.col("phase_id").cast(pl.Int64))
        self.temperature = add_temperature
        if just_mp:
            self.seebeck = pl.read_json(f"{path_mpds}mp_seebeck.json")
        else:
            self.seebeck = pl.read_json(f"{path_mpds}median_seebeck.json")
        data = make_normalization(self.seebeck.join(self.poly, on="phase_id", how="inner"))
        self.data = [list(data.row(i)) for i in range(len(data))][:40]

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

        for i, d in enumerate(poly_el):
            x_vector.append([])
            x_vector[i].append(d)
            # if descriptor is vector of counts
            if len(poly_el) > len(poly_type):
                x_vector[i].append(poly_type[0])
            else:
                x_vector[i].append(poly_type[i])
            if len(poly) == 3:
                if self.temperature:
                    x_vector[i].append(t)
                else:
                    x_vector[i].append(poly_vertex[i])
            elif len(poly) == 4:
                x_vector[i].append(poly_vertex[i])
                x_vector[i].append(t)

        node_features = torch.tensor(x_vector)

        edge_index = []

        for i in range(len(poly_el)):
            for j in range(i + 1, len(poly_el)):
                # graph is undirected, so we duplicate edge
                if i != j:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        if edge_index == []:
            edge_index.append([i, i])

        graph_data = Data(
            x=node_features, edge_index=torch.tensor(edge_index).t().contiguous()
        )
        return graph_data
