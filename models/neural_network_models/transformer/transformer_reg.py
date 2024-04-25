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
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm

r2 = R2Score()
mean_absolute_error = MeanAbsoluteError()


class TransformerModel(nn.Module):
    """A transformer model. Contains an encoder (without decoder)"""

    def __init__(self, n_feature, heads):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_feature,
            nhead=heads,
            batch_first=True,
            activation="gelu",
            dropout=0,
            norm_first=True,
        )
        self.agg_token = torch.rand((1, 1, n_feature))
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, norm=None
        )
        self.layer1 = nn.Linear(n_feature, 4 * n_feature * heads)
        self.layer2 = nn.Linear(4 * n_feature * heads, 1 * n_feature)
        self.layer3 = nn.Linear(1 * n_feature, 1)
        self.activ = nn.ELU()

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

    def fit(self, model, optimizer, train_data: Subset) -> None:
        """Train model"""
        model.train()
        for epoch in tqdm(range(50)):
            mean_loss = 0
            cnt = 0
            for y, poly, p_type in train_data:
                data = [eval(poly), eval(p_type)]
                if len(data[0]) == 0:
                    continue
                cnt += 1
                optimizer.zero_grad()
                out = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                loss = F.mse_loss(out, torch.tensor(y))
                loss.backward()
                optimizer.step()
                mean_loss += loss

            print(f"--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------")

            if epoch % 1 == 0:
                torch.save(
                    model.state_dict(),
                    r"/root/projects/ml-selection/models/neural_network_models/transformer/weights/20_01.pth",
                )

    def val(self, model, test_data: Subset) -> None:
        """Test model"""

        model.eval()

        preds = []
        y_s = []

        with torch.no_grad():
            for y, poly, p_type in test_data:
                data = [eval(poly), eval(p_type)]
                if len(data[0]) == 0:
                    continue
                pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
                preds.append(pred)
                y_s.append(y)

        mean_absolute_error.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
        mae_result = mean_absolute_error.compute()

        r2.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
        r2_res = r2.compute()

        torch.save(
            model.state_dict(),
            r"/root/projects/ml-selection/models/neural_network_models/transformer/weights/20_01.pth",
        )

        print("R2: ", r2_res, " MAE: ", mae_result)


if __name__ == "__main__":
    poly = pd.read_csv(
        f"/root/projects/ml-selection/data/processed_data/large_poly_descriptor.csv",
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
    model = TransformerModel(n_feature=2, heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model.fit(model, optimizer, train_data)
    model.val(model, test_data)
