"""
Selection of hyperparameters for GCN.
"""

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError

from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset
from models.neural_network_models.GCN.gcn_regression_model import GCN


def create_params(seed: int = 1) -> tuple:
    """
    Create params:
    n_hidden, n_hidden2 - hidden layers in the specified range
    activation - activation function
    max_ep - total number of epoch
    """
    rnd = np.random.RandomState(seed)

    n_hidden = rnd.randint(1, 32)
    n_hidden2 = rnd.randint(1, 32)
    activation = rnd.choice(["leaky_relu", "relu", "elu", "tanh"])

    lr = rnd.uniform(low=0.001, high=0.01)
    max_ep = rnd.randint(3, 25)

    return (n_hidden, n_hidden2, activation, lr, max_ep)


def search_params(test_dataloader: DataLoader, train_dataloader: DataLoader) -> list:
    """
    Train and test on visible hyperparameters, measure metrics: R2, MAE
    """
    max_trials = 25
    results = []

    for i in range(max_trials):
        print("Search trial " + str(i + 1))
        (n_hidden, n_hidden2, activation, lr, max_ep) = create_params(seed=i * 24)

        results.append([[n_hidden, n_hidden2, activation, lr, max_ep]])
        print((n_hidden, n_hidden2, activation, lr, max_ep))

        device = torch.device("cpu")
        model = GCN(n_hidden, n_hidden2, activation).to(device)

        model.train()
        model.fit(model, max_ep, train_dataloader, lr=lr, device=device)

        model.eval()
        r2_res, mae_result = model.val(model, test_dataloader, device)
        results[i].append([mae_result, r2_res])
        print(f"MAE: {mae_result}, R2: {r2_res}")

    return results


def main():
    print("\nBegin hyperparameter random search")

    dataset = CrystalGraphVectorsDataset()

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

    result = search_params(train_dataloader, test_dataloader)

    for res in result:
        print(res)


if __name__ == "__main__":
    main()
