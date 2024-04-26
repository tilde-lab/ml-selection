import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm

r2 = R2Score()
mae = MeanAbsoluteError()


class CNNModel(torch.nn.Module):
    """Convolutional Network"""
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=0
        )
        self.cnn2 = torch.nn.Conv2d(
            in_channels=32, out_channels=4, kernel_size=2, stride=1, padding=0
        )
        self.activ = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(392, 1)

    def forward(self, x):
        out = self.cnn1(x.type(torch.float))
        out = self.activ(out)

        out = self.cnn2(out)
        out = self.activ(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

    def fit(
        self,
        model,
        ep: int,
        train_dataloader: DataLoader,
        optimizer: torch.optim,
        device: torch.device,
    ) -> None:
        """
        Train model
        """
        model.train()
        for epoch in tqdm(range(ep)):
            mean_loss = 0
            cnt = 0
            for data in train_dataloader:
                y, x1, x2, x3 = data
                x = torch.cat(
                    (
                        torch.tensor([eval(i) for i in x1]).unsqueeze(-1),
                        torch.tensor([eval(i) for i in x2]).unsqueeze(-1),
                        torch.tensor([eval(i) for i in x3]).unsqueeze(-1),
                    ),
                    dim=2,
                )
                x = torch.unsqueeze(x, 0).unsqueeze(-3).squeeze(0)
                cnt += 1
                optimizer.zero_grad()
                out = model(x.to(device))
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
                r"/root/projects/ml-selection/models/neural_network_models/CNN/20_02.pth",
            )

    def val(
        self, model, test_dataloader: DataLoader, device: torch.device
    ) -> None:
        """Test model"""

        model.eval()
        with torch.no_grad():
            cnt = 0
            for data in test_dataloader:
                y, x1, x2, x3 = data
                x = torch.cat(
                    (
                        torch.tensor([eval(i) for i in x1]).unsqueeze(-1),
                        torch.tensor([eval(i) for i in x2]).unsqueeze(-1),
                        torch.tensor([eval(i) for i in x3]).unsqueeze(-1),
                    ),
                    dim=2,
                )
                x = torch.unsqueeze(x, 0).unsqueeze(-3).squeeze(0)
                cnt += 1
                pred = model(x.to(device))
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


if __name__ == "__main__":
    poly = pd.read_csv(
        f"/root/projects/ml-selection/data/processed_data/3_features_poly_descriptor.csv",
    )
    seebeck = pd.read_json(
        "/root/projects/ml-selection/data/raw_data/median_seebeck.json",
        orient="split",
    )
    dataset = (
        pd.merge(seebeck, poly, on="phase_id", how="inner")
        .drop(columns=["phase_id", "Formula"])
        .values.tolist()
    )

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(
        dataset, range(train_size, train_size + test_size)
    )
    train_dataloader = DataLoader(
        train_data, batch_size=4000, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(
        test_data, batch_size=1000, shuffle=False, num_workers=0
    )

    device = torch.device("cpu")
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=5e-4)

    model.fit(
        model,
        50,
        train_dataloader,
        optimizer,
        device,
    )
    model.val(model, test_dataloader, device)

    torch.save(
        model.state_dict(),
        r"/root/projects/ml-selection/models/neural_network_models/CNN/20_02.pth",
    )
