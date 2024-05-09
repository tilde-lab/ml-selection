"""
Training a TransformerModel to predict the Seebeck value.

Transformer made from encoder (without decoder). Uses token-vector to represent embedding for each crystal.
Tensor of tokens is fed to fully connected layer. Next, loss is calculated as in standard models.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Subset
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm

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
            activation='gelu',
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

    def fit(self, model, optimizer, ep, train_data: Subset) -> None:
        """Train model"""
        model.train()
        for epoch in tqdm(range(ep)):
            mean_loss = 0
            cnt = 0
            if self.n_feature == 4:
                for y, poly, p_vert, p_type, temp in train_data:
                    data = [eval(poly), eval(p_vert), eval(p_type), [temp]*len(eval(poly))]
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
                        data = [eval(poly), eval(p_vert), eval(p_type)]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]]*118
                    except:
                        data = [eval(poly), eval(p_vert), [p_type]*len(eval(poly))]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]]*118
                    cnt += 1
                    optimizer.zero_grad()
                    out = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    loss = F.mse_loss(out, torch.tensor(y))
                    loss.backward()
                    optimizer.step()
                    mean_loss += loss
            if self.n_feature == 2:
                for y, els, p_type in train_data:
                    data = [eval(els), eval(p_type)]
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

    def val(self, model, test_data: Subset) -> None:
        """Test model"""

        model.eval()

        r2.reset()
        mae.reset()

        preds = []
        y_s = []

        with torch.no_grad():
            if self.n_feature == 2:
                for y, els, p_type in test_data:
                    data = [eval(els), eval(p_type)]
                    if len(data[0]) != len(data[1]):
                        while len(data[0]) != len(data[1]):
                            data[1].append(data[1][0])
                    pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    preds.append(pred)
                    y_s.append(y)
            if self.n_feature == 3:
                for y, poly, p_vertex, p_type in test_data:
                    try:
                        data = [eval(poly), eval(p_vertex), eval(p_type)]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]] * 118
                    except:
                        data = [eval(poly), eval(p_vertex), [p_type] * len(eval(poly))]
                        if len(data[0]) == 118:
                            data[1] = [data[1][0]] * 118
                    pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    preds.append(pred)
                    y_s.append(y)
            if self.n_feature == 4:
                for y, poly, p_vert, p_type, temp in test_data:
                    data = [eval(poly), eval(p_vert), eval(p_type), [temp]*len(eval(poly))]
                    if len(data[0]) != len(data[1]):
                        while len(data[0]) != len(data[1]):
                            data[1].append(data[1][0])
                    pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                    preds.append(pred)
                    y_s.append(y)

        mae.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
        mae_result = mae.compute()

        r2.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
        r2_res = r2.compute()

        mape.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
        mape_res = mape.compute()

        print(
            "R2: ",
            r2_res,
            " MAE: ",
            mae_result,
            " MAPE: ",
            mape_res,
            " Pred from",
            min(preds),
            " to ",
            max(preds),
        )

        torch.save(
            model.state_dict(),
            r"/root/projects/ml-selection/models/neural_network_models/transformer/weights/0001.pth",
        )

        return [r2_res, mae_result]


if __name__ == "__main__":
    poly = pd.read_csv(
        f"/root/projects/ml-selection/data/processed_data/poly/3_features.csv",
    )
    seebeck = pd.read_json(
        "/root/projects/ml-selection/data/raw_data/median_seebeck.json", orient='split',
    )
    dataset = pd.merge(seebeck, poly, on="phase_id", how="inner").drop(columns=['phase_id', 'Formula']).values.tolist()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(
        dataset, range(train_size, train_size + test_size)
    )
    model = TransformerModel(4, 4, 16, 'elu')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006479739574204421, weight_decay=5e-4)

    model.fit(model, optimizer, 5, train_data)
    model.val(model, test_data)
