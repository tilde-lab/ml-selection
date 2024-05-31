import torch
import torch.nn.functional as F
from sklearn.metrics import explained_variance_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm
import pickle
import yaml
import numpy as np

from data.poly_store import get_poly_info
from data_massage.metrics.statistic_metrics import theils_u
from datasets.poly_graph_dataset import PolyGraphDataset

r2 = R2Score()
mae = MeanAbsoluteError()
mape = MeanAbsolutePercentageError()

with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
    yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
    path_sc = yaml_f["scaler_path"]

class GCN(torch.nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, features, n_hidden=16, n_hidden2=32, activation="elu"):
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
        name_to_save="gcn_w",
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
                print(
                    f"--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------"
                )
            if epoch % 5 == 0:
                torch.save(
                    model.state_dict(),
                    f"/root/projects/ml-selection/models/neural_network_models/GCN/weights/{name_to_save}.pth",
                )

    def val(
        self,
        model,
        test_dataloader: DataLoader,
        device: torch.device,
        name_to_save="gcn_w",
    ) -> torch.Tensor:
        """Test model"""
        (model.eval(), r2.reset(), mae.reset())

        preds = None
        with open(f'{path_sc}scalerSeebeck coefficient.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with torch.no_grad():
            for data, y in test_dataloader:
                pred = model(data.to(device))
                pred, y = torch.tensor(scaler.inverse_transform(np.array(pred))), torch.tensor(scaler.inverse_transform(np.array(y).reshape(-1, 1)))

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
            "Theil's U: ",
            theils_u_res,
            " Pred from",
            pred.min(),
            " to ",
            pred.max(),
        )

        torch.save(
            model.state_dict(),
            f"/root/projects/ml-selection/models/neural_network_models/GCN/weights/{name_to_save}.pth",
        )

        return r2_res, mae_result


def main(epoch=5, device="cpu", name_to_save="w_gcn", batch_size=2):
    def get_ds(path, f, temperature):
        dataset = PolyGraphDataset(path, f, temperature)
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
            train_dataloader, test_dataloader = get_ds(
                path, len(features[idx]), temperature
            )

            device = torch.device(device)
            model = GCN(len(features[idx]), 16, 32, "relu").to(device)
            try:
                model.load_state_dict(
                    torch.load(
                        f"/root/projects/ml-selection/models/neural_network_models/GCN/weights/{name_to_save}_{len(features[idx])}.pth"
                    )
                )
            except:
                pass

            model.fit(
                model,
                epoch,
                train_dataloader,
                device,
                lr=0.008598391737229157,
                name_to_save=name_to_save + str(len(features[idx])),
            )
            model.val(
                model,
                test_dataloader,
                device,
                name_to_save=name_to_save + str(len(features[idx])),
            )

    for k, features in enumerate(total_features):
        if k == 1:
            temperature, feature = True, 3
        else:
            temperature, feature = False, 2

        train_dataloader, test_dataloader = get_ds(
            poly_just_graph_models, feature, temperature
        )
        device = torch.device(device)
        model = GCN(feature, 16, 32, "tanh").to(device)
        try:
            model.load_state_dict(
                torch.load(
                    f"/root/projects/ml-selection/models/neural_network_models/GCN/weights/{name_to_save}_{features}.pth"
                )
            )
        except:
            pass

        model.fit(
            model,
            epoch,
            train_dataloader,
            device,
            lr=0.008598391737229157,
            name_to_save=name_to_save + str(feature),
        )
        model.val(
            model, test_dataloader, device, name_to_save=name_to_save + str(feature)
        )


if __name__ == "__main__":
    main(epoch=1, name_to_save="31_05")
