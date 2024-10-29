"""
Test GAT on CrystalGraphVectorsDataset
"""

import torch
from datasets.poly_graph_dataset import PolyGraphDataset
from models.neural_network_models.GAT.gat_regression_model import GAT
from torch_geometric.loader import DataLoader

path = "ml_selection/data/processed_data/poly/0_features.json"
n_features = 2
temperature = False

dataset = PolyGraphDataset(path, n_features, temperature)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=0)

device = torch.device("cpu")
model = GAT(features=n_features).to(device)

model.load_state_dict(
    torch.load(
        r"ml_selection/models/neural_network_models/GAT/weights/20_01.pth"
    )
)

model.val(model, test_dataloader, device)
