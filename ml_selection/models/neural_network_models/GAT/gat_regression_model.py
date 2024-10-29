import pickle

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from data.poly_store import get_poly_info
from datasets.poly_graph_dataset import PolyGraphDataset
from metrics.statistic_metrics import theils_u
from sklearn.metrics import explained_variance_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, Linear
from torch_geometric.utils import scatter
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm

CONF = "ml_selection/configs/config.yaml"

with open(CONF, "r") as yamlfile:
    yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
    PATH_SC = yaml_f["scaler_path"]
    WEIGHTS_DIR = yaml_f["weights"]

r2 = R2Score()
mae = MeanAbsoluteError()
mape = MeanAbsolutePercentageError()


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, features, hidden=32, hidden2=16, activation="relu"):
        super().__init__()
        self.conv1 = GATv2Conv(features, hidden, 1, edge_dim=1)
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
        Forward pass

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
            # is CrystalVectorsGraphDataset or PolyGraphDataset
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
        self,
        model,
        ep: int,
        train_dataloader: DataLoader,
        optimizer: torch.optim,
        device: torch.device,
        save_dir: str = "./gat.pth",
    ) -> None:
        """
        Train model

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
                mape.update(out.reshape(-1), y)
            print(f"--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------")
            if epoch % 5 == 0:
                torch.save(model.state_dict(), save_dir)

    def val(
        self,
        model,
        test_dataloader: DataLoader,
        device: torch.device,
        save_dir="./gat.pth",
    ) -> torch.Tensor:
        """Test model"""
        (model.eval(), r2.reset(), mae.reset())

        preds = None
        with open(f"{PATH_SC}scalerSeebeck coefficient.pkl", "rb") as f:
            scaler = pickle.load(f)

        with torch.no_grad():
            for data, y in test_dataloader:
                pred = model(data.to(device))
                pred, y = (
                    torch.tensor(scaler.inverse_transform(np.array(pred.cpu()))),
                    torch.tensor(
                        scaler.inverse_transform(np.array(y.cpu()).reshape(-1, 1))
                    ),
                )

                if preds != None:
                    preds = torch.cat((preds, pred), dim=0)
                    y_true = torch.cat((y_true, y), dim=0)
                else:
                    preds, y_true = pred, y
                mae.update(pred, y)
                r2.update(pred, y)

        mae_result = mae.compute()
        r2_res = r2.compute()
        evs = explained_variance_score(preds, y_true)
        theils_u_res = theils_u(preds, y_true)

        print(
            "R2: ",
            r2_res,
            " MAE: ",
            mae_result,
            " EVS: ",
            evs,
            "Theil's U:",
            theils_u_res,
            " Pred from",
            pred.min(),
            " to ",
            pred.max(),
        )
        torch.save(model.state_dict(), save_dir)

        return r2_res, mae_result


def main(epoch=5, device="cpu", name_to_save="w_gat", batch_size=2, just_mp=False):
    def get_ds(path, f, temperature):
        dataset = PolyGraphDataset(path, f, temperature, just_mp)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size

        train_data = torch.utils.data.Subset(dataset, range(train_size))
        test_data = torch.utils.data.Subset(
            dataset, range(train_size, train_size + test_size)
        )
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=0
        )
        return train_dataloader, test_dataloader

    (
        poly_dir_path,
        poly_path,
        poly_just_graph_models,
        poly_features,
        poly_temperature_features,
    ) = get_poly_info()

    if just_mp:
        for i in range(len(poly_path)):
            poly_path[i] = poly_path[i].replace(".json", "_mp.json")

    total_features = []
    (
        total_features.append(poly_features),
        total_features.append(poly_temperature_features),
    )

    for k, features in enumerate(total_features):
        if k == 1:
            temperature = True
        else:
            temperature = False
        for idx, path in enumerate(poly_path):
            path_to_w = (
                WEIGHTS_DIR
                + f"gat_{name_to_save}_{len(features[idx])}_{temperature}.pth"
            )
            train_dataloader, test_dataloader = get_ds(
                path, len(features[idx]), temperature
            )

            device = torch.device(device)
            model = GAT(len(features[idx]), 16, 32, "tanh").to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.008598391737229157, weight_decay=5e-4
            )

            try:
                model.load_state_dict(torch.load(path_to_w))
                print("Successfully loaded pretrained weights to GAT")
            except:
                print("No pretrained weights found for GAT")

            model.fit(
                model, epoch, train_dataloader, optimizer, device, save_dir=path_to_w
            )
            model.val(model, test_dataloader, device, save_dir=path_to_w)


if __name__ == "__main__":
    main(epoch=1, name_to_save="31_05")
