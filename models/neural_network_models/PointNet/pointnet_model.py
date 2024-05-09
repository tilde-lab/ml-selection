from torch import Tensor
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool
from datasets.point_cloud_dataset import PointCloudDataset
from tqdm import tqdm


r2 = R2Score()
mae = MeanAbsoluteError()
mape = MeanAbsolutePercentageError()


class PointNetLayer(MessagePassing):
    """PointNet encoder"""

    def __init__(self, in_channels: int, out_channels: int, d):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3 or 4 (if 4D)).
        self.mlp = Sequential(
            Linear(in_channels + d, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j: Tensor, pos_j: Tensor, pos_i: Tensor) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3 or 4]
        # pos_i: The central node position as shape [num_edges, 3 or 4]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNet(torch.nn.Module):
    """Point Network model"""

    def __init__(self, d, hidden=32):
        super().__init__()

        self.conv1 = PointNetLayer(d, hidden, d)
        self.conv2 = PointNetLayer(hidden, hidden, d)
        self.linear = Linear(hidden, 1)

    def forward(
        self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        return self.linear(h)


def train(model, ep, train_loader, optimizer):
    model.train()
    criterion = torch.nn.MSELoss()

    for e in range(ep):
        total_loss = 0
        for d in train_loader:
            data, y = d
            optimizer.zero_grad()
            logits = model(data.pos, data.edge_index.to(torch.int64), data.batch)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
            mape.update(logits.reshape(-1), y)
        f"--------Mean loss for epoch {e} is {total_loss / len(train_loader.dataset)}--------"
    return total_loss / len(train_loader.dataset)


def val(model, test_loader):
    model.eval()
    r2.reset()
    mae.reset()

    preds = []
    with torch.no_grad():
        for d in test_loader:
            data, y = d
            pred = model(data.pos, data.edge_index.to(torch.int64), data.batch)
            preds.append(pred)
            mae.update(pred.reshape(-1), y)
            r2.update(pred.reshape(-1), y)
            mape.update(pred.reshape(-1), y)

        mae_result = mae.compute()
        r2_res = r2.compute()
        mape_res = mape.compute()

        print(
            "R2: ",
            r2_res,
            " MAE: ",
            mae_result,
            " MAPE: ",
            mape_res,
            "Pred from",
            min([i.min() for i in preds]),
            " to ",
            max([i.max() for i in preds]),
        )
    return r2_res, mae_result


if __name__ == "__main__":
    dataset = PointCloudDataset()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(
        dataset, range(train_size, train_size + test_size)
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    model = PointNet(4)
    model.load_state_dict(
        torch.load(
            r"/models/neural_network_models/weights/30_01.pth"
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in tqdm(range(1)):
        loss = train(model, 5, train_loader, optimizer)
        test_acc = val(model, test_loader)
