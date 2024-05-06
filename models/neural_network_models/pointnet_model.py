from torch import Tensor
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool
from datasets.point_cloud_dataset import PointCloudDataset
from tqdm import tqdm


r2 = R2Score()
mae = MeanAbsoluteError()


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(
            Linear(in_channels + 4, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(4, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.linear = Linear(32, 1)

    def forward(self,
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


model = PointNet()

if __name__ == '__main__':
    dataset = PointCloudDataset()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(
        dataset, range(train_size, train_size + test_size)
    )
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=64, shuffle=False, num_workers=0
    )

    model = PointNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    def train(ep):
        model.train()

        total_loss = 0
        for d in train_loader:
            data, y = d
            optimizer.zero_grad()
            logits = model(data.pos, data.edge_index.to(torch.int64), data.batch)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs

        print(f'Train loss for epoch {ep} is: ', total_loss / len(train_loader.dataset))
        torch.save(
            model.state_dict(),
            r"/root/projects/ml-selection/models/neural_network_models/30_01.pth",
        )
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def val():
        model.eval()
        r2.reset()
        mae.reset()
        cnt = 0

        for d in test_loader:
            data, y = d
            cnt += 1
            pred = model(data.pos, data.edge_index.to(torch.int64), data.batch)
            mae.update(pred.reshape(-1), y)
            r2.update(pred.reshape(-1), y)

        mae_result = mae.compute()
        r2_res = r2.compute()
        print(
            "R2: ",
            r2_res,
            " MAE: ",
            mae_result,
            "Pred from",
            pred.min(),
            " to ",
            pred.max(),
        )

    for epoch in tqdm(range(25)):
        loss = train(epoch)
        test_acc = val()
