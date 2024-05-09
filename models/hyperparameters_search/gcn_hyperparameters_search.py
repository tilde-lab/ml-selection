"""
Selection of hyperparameters for GCN.
"""

import torch
from torch_geometric.loader import DataLoader
import optuna

from datasets.poly_graph_dataset import PolyGraphDataset
from models.neural_network_models.GCN.gcn_regression_model import GCN

BEST_WEIGHTS = None
BEST_R2 = -100


def main(path: str, features: int, ds: int, temperature: bool):
    if temperature:
        features += 1

    def objective(trial) -> int:
        """Search of hyperparameters"""
        global BEST_WEIGHTS, BEST_R2
        dataset = PolyGraphDataset(path, features, temperature)

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
            test_data, batch_size=5240, shuffle=False, num_workers=0
        )


        device = torch.device("cpu")

        hidden = trial.suggest_categorical("hidden", [8, 16, 32])
        hidden2 = trial.suggest_categorical("hidden2", [8, 16, 32, 64])
        lr = trial.suggest_float("lr", 0.0001, 0.01)
        ep = trial.suggest_int("ep", 4, 10)
        activ = trial.suggest_categorical("activ", ["leaky_relu", "relu", "elu", "tanh"])

        model = GCN(features, hidden, hidden2, activation=activ).to(device)

        # train and test
        model.fit(model, ep, train_dataloader, device, lr=lr)
        r2, mae = model.val(model, test_dataloader, device)

        if r2 > BEST_R2:
            BEST_R2 = r2
            BEST_WEIGHTS = model.state_dict()

        return r2

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=5)

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
        torch.save(BEST_WEIGHTS, f'best_gcn_weights{ds}.pth')

    return res


if __name__ == "__main__":
    path = '/root/projects/ml-selection/data/processed_data/poly/0_features.csv'
    features = 2
    temperature = False
    main(path, features, 1, temperature)
