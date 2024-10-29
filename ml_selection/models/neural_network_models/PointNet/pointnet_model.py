import torch
import yaml
from datasets.point_cloud_dataset import PointCloudDataset
from metrics.statistic_metrics import theils_u
from sklearn.metrics import explained_variance_score
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_max_pool
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError

CONF = "ml_selection/configs/config.yaml"

with open(CONF, "r") as yamlfile:
    yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
    WEIGHTS_DIR = yaml_f["weights"]

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
        print(
            f"--------Mean loss for epoch {e} is {total_loss / len(train_loader.dataset)}--------"
        )
    return total_loss / len(train_loader.dataset)


def val(model, test_loader, save_dir):
    model.eval(), r2.reset(), mae.reset()

    preds = None
    with torch.no_grad():
        for d in test_loader:
            data, y = d
            pred = model(data.pos, data.edge_index.to(torch.int64), data.batch)
            if preds != None:
                preds = torch.cat((preds, pred), dim=0)
                y_true = torch.cat((y_true, y), dim=0)
            else:
                preds, y_true = pred, y
            mae.update(pred.reshape(-1), y)
            r2.update(pred.reshape(-1), y)

        mae_result = mae.compute()
        r2_res = r2.compute()
        evs = explained_variance_score(preds, y_true)
        theils_u_res = theils_u(preds, y_true)

        torch.save(model.state_dict(), save_dir)

        print(
            "R2: ",
            r2_res,
            " MAE: ",
            mae_result,
            " EVS: ",
            evs,
            "Theil's U: ",
            theils_u_res,
            "Pred from",
            min([i.min() for i in preds]),
            " to ",
            max([i.max() for i in preds]),
        )
    return r2_res, mae_result


def main(
    epoch: int = 20, batch_size: int = 2, name_to_save="w_pn", just_mp: bool = False
):
    features = [3, 4]

    for f in features:
        path_to_w = WEIGHTS_DIR + f"pointnet_{name_to_save}_{f}.pth"
        dataset = PointCloudDataset(features=f, just_mp=just_mp)

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size

        train_data = torch.utils.data.Subset(dataset, range(train_size))
        test_data = torch.utils.data.Subset(
            dataset, range(train_size, train_size + test_size)
        )

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=0
        )

        model = PointNet(f)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        try:
            model.load_state_dict(torch.load(path_to_w))
            print("Successfully loaded pretrained weights to PointNet")
        except:
            print("No pretrained weights found for PointNet")

        _ = train(model, epoch, train_loader, optimizer)
        _ = val(model, test_loader, save_dir=path_to_w)


if __name__ == "__main__":
    main(name_to_save="31_05", just_mp=True)
