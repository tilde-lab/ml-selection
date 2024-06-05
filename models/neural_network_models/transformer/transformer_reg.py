"""
Training a TransformerModel to predict the Seebeck value.

Transformer made from encoder (without decoder). Uses token-vector to represent embedding for each crystal.
Tensor of tokens is fed to fully connected layer. Next, loss is calculated as in standard models.
"""
from data_massage.normalization.normalization import make_normalization
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import explained_variance_score
from torch.utils.data import Subset
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm
import pickle
import yaml
import numpy as np

from data.poly_store import get_poly_info
from data_massage.metrics.statistic_metrics import theils_u


with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
    yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
    path_sc = yaml_f["scaler_path"]

r2 = R2Score()
mae = MeanAbsoluteError()
mape = MeanAbsolutePercentageError()


class TransformerModel(nn.Module):
    """A transformer model. Contains an encoder (without decoder)"""

    def __init__(self, n_feature, heads, hidd, activation):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_feature,
            nhead=heads,
            batch_first=True,
            activation="gelu",
            dropout=0,
            norm_first=True,
        )
        self.n_feature = n_feature
        self.agg_token = torch.rand((1, 1, n_feature))
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, norm=None
        )
        self.layer1 = nn.Linear(n_feature, n_feature * hidd * heads)
        self.layer2 = nn.Linear(hidd * n_feature * heads, 1 * hidd)
        self.layer3 = nn.Linear(hidd, 1)
        if activation == "elu":
            self.activ = F.elu
        elif activation == "relu":
            self.activ = F.relu
        elif activation == "leaky_relu":
            self.activ = F.leaky_relu
        elif activation == "tanh":
            self.activ = F.tanh

    def forward(self, batch: list) -> torch.Tensor:
        """
        Forward pass.
        'agg_token' is concatenated to every matrix of crystal. Feeding into the transformer
        occurs separately for each crystal. Before transfer to fully connected layer, token embeddings
        are extracted

        Parameters
        ----------
        data : list
            list with torch.Tensor. Consist of just 1 vectors structure
        """
        emb_list = []
        for data in batch:
            x = torch.cat([self.agg_token, data], dim=1)
            x = self.transformer_encoder(x)

            # get token embedding
            token_emb = x[:, 0]
            emb_list.append(token_emb)

        x = self.layer1(emb_list[0])
        x = self.activ(x)
        x = self.layer2(x)
        x = self.activ(x)
        x = self.layer3(x)
        return x

    def fit(
        self, model, optimizer, ep, train_data: Subset, name_to_save="tran_w"
    ) -> None:
        """Train model"""
        model.train()
        for epoch in tqdm(range(ep)):
            mean_loss = 0
            cnt = 0
            if self.n_feature == 4:
                for y, poly, p_vert, p_type, temp in train_data:
                    data = [poly, p_vert, p_type, [temp] * len(poly)]
                    cnt += 1
                    optimizer.zero_grad()
                    out = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    loss = F.mse_loss(out, torch.tensor(y))
                    loss.backward()
                    optimizer.step()
                    mean_loss += loss
            if self.n_feature == 3:
                for y, poly, p_vert, p_type in train_data:
                    try:
                        data = [poly, p_vert, p_type]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]] * 118
                            data[2] = [data[2]] * 118
                    except:
                        data = [poly, p_vert, [p_type] * len(poly)]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]] * 118
                            data[2] = [data[2]] * 118
                    if type(data[2]) != list and len(data[0]) == 100:
                        data[2] = [data[2]] * 100
                    cnt += 1
                    optimizer.zero_grad()
                    out = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    loss = F.mse_loss(out, torch.tensor(y))
                    loss.backward()
                    optimizer.step()
                    mean_loss += loss
            if self.n_feature == 2:
                for y, els, p_type in train_data:
                    data = [els, p_type]
                    if len(data[0]) != len(data[1]):
                        while len(data[0]) != len(data[1]):
                            data[1].append(data[1][0])
                    cnt += 1
                    optimizer.zero_grad()
                    out = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    loss = F.mse_loss(out, torch.tensor(y))
                    loss.backward()
                    optimizer.step()
                    mean_loss += loss

            print(f"--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------")

    def val(self, model, test_data: Subset, f: int = 0, name_to_save="tran_w", temperature=True,) -> None:
        """Test model"""
        (model.eval(), r2.reset(), mae.reset())

        preds, y_s = [], []
        with open(f'{path_sc}scalerSeebeck coefficient.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with torch.no_grad():
            if self.n_feature == 2:
                for y, els, p_type in test_data:
                    data = [els, p_type]
                    if len(data[0]) != len(data[1]):
                        while len(data[0]) != len(data[1]):
                            data[1].append(data[1][0])
                    pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    pred, y = torch.tensor(scaler.inverse_transform(np.array(pred))), torch.tensor(
                        scaler.inverse_transform(np.array(y).reshape(-1, 1))[0])

                    preds.append(pred)
                    y_s.append(y)
            if self.n_feature == 3:
                for y, poly, p_vertex, p_type in test_data:
                    try:
                        data = [poly, p_vertex, p_type]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]] * 118
                    except:
                        data = [poly, p_vertex, [p_type] * len(poly)]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]] * 118
                    if type(data[2]) != list and len(data[0]) == 100:
                        data[2] = [data[2]] * 100
                    elif type(data[2]) != list and len(data[0]) == 118:
                        data[2] = [data[2]] * 118
                    pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    pred, y = torch.tensor(scaler.inverse_transform(np.array(pred))), torch.tensor(
                        scaler.inverse_transform(np.array(y).reshape(-1, 1))[0])
                    preds.append(pred)
                    y_s.append(y)
            if self.n_feature == 4:
                for y, poly, p_vert, p_type, temp in test_data:
                    data = [poly, p_vert, p_type, [temp] * len(poly)]
                    if len(data[0]) != len(data[1]):
                        while len(data[0]) != len(data[1]):
                            data[1].append(data[1][0])
                    pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    pred, y = torch.tensor(scaler.inverse_transform(np.array(pred))), torch.tensor(
                        scaler.inverse_transform(np.array(y).reshape(-1, 1))[0])
                    preds.append(pred), y_s.append(y)

        mae.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
        mae_result = mae.compute()

        r2.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
        r2_res = r2.compute()

        evs = explained_variance_score(
            np.array([i[0][0] for i in preds]), np.array(y_s)
        )
        theils_u_res = theils_u(np.array([i[0][0] for i in preds]), np.array(y_s))

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
            min(preds),
            " to ",
            max(preds),
        )

        torch.save(
            model.state_dict(),
            f"/root/projects/ml-selection/models/neural_network_models/transformer"
            f"/weights/{name_to_save}_{f}_{temperature}.pth",
        )
        print(f'Weights saved with name: {name_to_save}_{f}_{temperature}.pth')

        return r2_res, mae_result


def main(epoch=5, name_to_save="tran_w", just_mp=False):
    def get_ds(poly_path, temperature):
        poly = pl.read_json(poly_path)
        if just_mp:
            seebeck = pl.read_json(
                "/root/projects/ml-selection/data/raw_mpds/mp_seebeck.json"
            )
        else:
            seebeck = pl.read_json(
                "/root/projects/ml-selection/data/raw_mpds/median_seebeck.json"
            )
        poly = poly.with_columns(pl.col("phase_id").cast(pl.Int64))
        dataset = make_normalization(seebeck.join(poly, on="phase_id", how="inner").drop(
            ["phase_id", "Formula"]
        ))
        if not (temperature):
            dataset = dataset.drop(columns=["temperature"])
        dataset = [list(dataset.row(i)) for i in range(len(dataset))]

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size

        train_data = torch.utils.data.Subset(dataset, range(train_size))
        test_data = torch.utils.data.Subset(
            dataset, range(train_size, train_size + test_size)
        )
        return train_data, test_data

    (
        poly_dir_path,
        poly_path,
        _,
        poly_features,
        poly_temperature_features,
    ) = get_poly_info()

    if just_mp:
        for i in range(len(poly_path)):
            poly_path[i] = poly_path[i].replace('.json', '_mp.json')

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
            train_dataloader, test_dataloader = get_ds(path, temperature)

            model = TransformerModel(len(features[idx]), len(features[idx]), 32, "tanh")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.0006479739574204421, weight_decay=5e-4
            )
            try:
                model.load_state_dict(
                    torch.load(
                        f"/root/projects/ml-selection/models/neural_network_models/transformer"
                        f"/weights/{name_to_save}_{len(features[idx])}_{temperature}.pth"
                    )
                )
                print('Successfully loaded pretrained weights to Transformer')
            except:
                print('No pretrained weights found for Transformer')

            model.fit(
                model,
                optimizer,
                epoch,
                train_dataloader,
                name_to_save=name_to_save + str(len(features[idx]))
            )
            model.val(
                model,
                test_dataloader,
                name_to_save=name_to_save + str(len(features[idx])),
                temperature=temperature,
                f=len(features[idx])
            )


if __name__ == "__main__":
    main(epoch=10, name_to_save='05_06', just_mp=True)
