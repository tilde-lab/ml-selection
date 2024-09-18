"""
Test GCN on CrystalGraphVectorsDataset
"""

import torch
from torch_geometric.loader import DataLoader

from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset
from models.neural_network_models.GCN.gcn_regression_model import GCN

dataset = CrystalGraphVectorsDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=0)

device = torch.device("cpu")
model = GCN(13, 16, "relu").to(device)

model.load_state_dict(
    torch.load(
        r"/root/projects/ml-selection/models/neural_network_models/GCN/weights/01.pth"
    )
)

model.val(model, test_dataloader, device)
