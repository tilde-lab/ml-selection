import polars as pd
import torch
from ase.data import chemical_symbols
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data.mendeleev_table import get_periodic_number


class CrystalGraphDataset(Dataset):
    """
    Dataset that represents each structure as a graph.
    Atom = graph node, abstract connection in a crystal = edge.
    For each node, features are periodic number of the atom and 3 coordinates.
    """

    def __init__(self):
        super().__init__()
        self.transform = self.build_graph
        self.file_path = "/root/projects/ml-selection/data/processed_data/total.csv"
        self.data_csv = pd.read_csv(self.file_path)
        self.data = self.data_csv.values.tolist()

    def __len__(self):
        """Return number of samples"""
        return len(self.data)

    def __getitem__(self, idx: int) -> [Data, float]:
        """Return one sample"""
        graph, seebeck = self.transform(self.data[idx])
        return [graph, seebeck]

    def atom_to_ordinal(self, atom: str):
        """
        Return ordinal number for specific atom.
        """
        return chemical_symbols.index(atom)

    def build_graph(self, mol_data: list) -> [Data, float]:
        """
        Make graph.
        Save atom coordinates as node attributes, calculates the length of edges (distance between atoms).
        Graph is fully connected - all atoms are connected by edges

        Parameters
        ----------
        mol_data : list
            Data with next values:
            ['phase_id', 'Formula', 'Seebeck coefficient', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']
        """
        ph, formula, seebeck, cell_abc_str, sg_n, basis_noneq, els_noneq = mol_data
        els_noneq = eval(els_noneq)
        basis_noneq = eval(basis_noneq)

        # create list with features for every node
        x_vector = [[get_periodic_number(atom)] for atom in els_noneq]

        # add coordinates to every node
        for i, atom in enumerate(els_noneq):
            x_vector[i].append(basis_noneq[i][0])
            x_vector[i].append(basis_noneq[i][1])
            x_vector[i].append(basis_noneq[i][2])

        node_features = torch.tensor(x_vector)

        edge_index = []
        edge_attr = []

        # to calculate distance between all atoms
        for i in range(len(basis_noneq)):
            for j in range(i + 1, len(basis_noneq)):
                distance = torch.norm(
                    torch.tensor(basis_noneq[i]) - torch.tensor(basis_noneq)
                )

                # graph is undirected, so we duplicate edge
                edge_index.append([i, j])
                edge_index.append([j, i])

                edge_attr.append(distance)
                edge_attr.append(distance)

        graph_data = Data(
            x=node_features,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_attr=torch.tensor(edge_attr),
        )
        return [graph_data, seebeck]
