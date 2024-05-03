import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm

from datasets.poly_graph_dataset import PolyGraphDataset

r2 = R2Score()
mae = MeanAbsoluteError()


class GCN(torch.nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, features, n_hidden, n_hidden2=4, activation="elu"):
        super().__init__()
        self.conv1 = GCNConv(features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden2)
        self.layer3 = Linear(n_hidden2, 1)
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
        Forward pass

        Parameters
        ----------
        data : DataBatch
            data from CrystalVectorsGraphDataset
        """
        x, edge_index = data.x, data.edge_index.type(torch.int64)

        x = self.conv1(x.float(), edge_index)
        x = self.activ(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.activ(x)

        x = scatter(x, data.batch, dim=0, reduce="mean")

        x = self.layer3(x)
        return x

    def fit(
        self,
        model,
        ep: int,
        train_dataloader: DataLoader,
        device: torch.device,
        lr=0.005,
    ):
        """Train model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

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
            if epoch % 1 == 0:
                torch.save(
                    model.state_dict(),
                    r"/root/projects/ml-selection/models/neural_network_models/GCN/weights/30_01.pth",
                )

    def val(
        self, model, test_dataloader: DataLoader, device: torch.device
    ) -> torch.Tensor:
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
    n_features = 2
    dataset = PolyGraphDataset(
        '/root/projects/ml-selection/data/processed_data/poly/poly_vector_of_count.csv',
        n_features
    )

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(
        dataset, range(train_size, train_size + test_size)
    )
    train_dataloader = DataLoader(
        train_data, batch_size=64, shuffle=False, num_workers=0
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1000, shuffle=False, num_workers=0
    )

    device = torch.device("cpu")
    model = GCN(n_features, 13, 16, "relu").to(device)

    model.fit(model, 1, train_dataloader, device, lr=0.008598391737229157)
    model.val(model, test_dataloader, device)


