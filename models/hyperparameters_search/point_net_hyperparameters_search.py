"""
Selection of hyperparameters for PointNetwork
"""

import optuna
import torch
from torch_geometric.loader import DataLoader

from datasets.point_cloud_dataset import PointCloudDataset
from models.neural_network_models.PointNet.pointnet_model import (PointNet,
                                                                  train, val)

BEST_WEIGHTS = None
BEST_R2 = -100


def main(features, ds):
    def objective(trial) -> int:
        """Search of hyperparameters"""
        global BEST_WEIGHTS, BEST_R2
        dataset = PointCloudDataset(features=features)

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
            test_data, batch_size=64, shuffle=False, num_workers=0
        )

        hidden = trial.suggest_categorical("hidden", [8, 16, 32, 64, 128, 256])
        lr = trial.suggest_float("lr", 0.0001, 0.01)
        ep = trial.suggest_int("ep", 1, 1)

        model = PointNet(features, hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train and test
        train(model, ep, train_dataloader, optimizer)
        r2, mae = val(model, test_dataloader)

        if r2 > BEST_R2:
            BEST_R2 = r2
            BEST_WEIGHTS = model.state_dict()

        return r2

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(), direction="maximize"
    )
    study.optimize(objective, n_trials=1)

    res = [study.best_trial]

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  R2: ", trial.values)
    print("  Params: ")

    parms = []
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        parms.append([key, value])
    res.append(parms)

    if BEST_WEIGHTS is not None:
        torch.save(BEST_WEIGHTS, f"best_pointnet_weights{ds}.pth")

    return res


if __name__ == "__main__":
    for idx, features in enumerate([3, 4]):
        main(features, idx)
