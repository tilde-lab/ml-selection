import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, Linear
from torch_geometric.utils import scatter
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm

from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset

r2 = R2Score()
mae = MeanAbsoluteError()


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, in_ch, hidden=8, hidden2=16, activation="relu"):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, hidden, 1, edge_dim=1)
        self.conv2 = GATv2Conv(hidden, hidden2, 1, edge_dim=1)
        self.layer3 = Linear(hidden2, 1)
        if activation == "elu":
            self.activ = F.elu
        elif activation == "relu":
            self.activ = F.relu
        elif activation == "leaky_relu":
            self.activ = F.leaky_relu
        elif activation == "tanh":
            self.activ = F.tanh

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.
        Parameters
        ----------
        data : DataBatch
            data from CrystalGraphDataset or CrystalVectorsGraphDataset, kind is automatically detected
        """
        edge_attr = None
        try:
            # is CrystalGraphDataset
            x, edge_index, edge_attr = (
                data.x,
                data.edge_index.type(torch.int64),
                data.edge_attr,
            )
        except:
            # is CrystalVectorsGraphDataset
            x, edge_index = data.x, data.edge_index.type(torch.int64)

        x = self.conv1(x.float(), edge_index=edge_index, edge_attr=edge_attr)
        x = self.activ(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.activ(x)

        x = scatter(x, data.batch, dim=0, reduce="mean")

        x = self.layer3(x)
        return x

    def fit(
            self, model, ep: int, train_dataloader: DataLoader, optimizer: torch.optim, device: torch.device
    ) -> None:
        """
        Train model.
        Parameters
        ----------
        model : GAT
            class instance
        ep : int
            number of training epochs
        train_dataloader : DataLoader
            class instance of torch_geometric.loader.DataLoader with training data
        optimizer : optimizer from Torch
            optimizer, for example torch.optim.Adam
        device : torch.device
            'cpu' or 'gpu'
        """
        model.train()
        for epoch in tqdm(range(ep)):
            mean_loss = 0
            cnt = 0
            for data, y in train_dataloader:
                cnt += 1
                optimizer.zero_grad()
                out = model(data.to(device))
                loss = F.mse_loss(out, y.to(device))
                loss.backward()
                optimizer.step()
                mean_loss += loss
                r2.update(out.reshape(-1), y)
            print(
                f"--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------R2 is {r2.compute()}"
            )
            torch.save(
                model.state_dict(),
                r"/root/projects/ml-selection/models/neural_network_models/GAT/weights/01.pth",
            )

    def val(self, model, test_dataloader: DataLoader, device: torch.device) -> torch.Tensor:
        """Test model"""

        model.eval()
        with torch.no_grad():
            cnt = 0
            for data, y in test_dataloader:
                cnt += 1
                pred = model(data.to(device))
                mae.update(pred.reshape(-1), y)
                mae_result = mae.compute()

                r2.update(pred.reshape(-1), y)
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

        return r2_res, mae_result


if __name__ == "__main__":
    # dataset with atoms and distance info
    dataset = CrystalGraphVectorsDataset()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(
        dataset, range(train_size, train_size + test_size)
    )
    train_dataloader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1000, shuffle=False, num_workers=0
    )

    device = torch.device("cpu")
    model = GAT(in_ch=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005919, weight_decay=5e-4)

    model.fit(model, 5, train_dataloader, optimizer, device, )
    model.val(model, test_dataloader, device)

    torch.save(
        model.state_dict(),
        r"/root/projects/ml-selection/models/neural_network_models/GAT/weights/01.pth",
    )
