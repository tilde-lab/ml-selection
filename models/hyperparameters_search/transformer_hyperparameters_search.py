"""
Selection of hyperparameters for GCN.
"""

import torch
import optuna
import pandas as pd

from models.neural_network_models.transformer.transformer_reg import TransformerModel

BEST_WEIGHTS = None
BEST_R2 = -100


def main(poly_path, features, ds):
    poly = pd.read_csv(poly_path)
    seebeck = pd.read_json(
        "/root/projects/ml-selection/data/raw_data/median_seebeck.json", orient='split',
    )
    dataset = pd.merge(
        seebeck, poly, on="phase_id", how="inner"
    ).drop(columns=['phase_id', 'Formula']).values.tolist()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(
        dataset, range(train_size, train_size + test_size)
    )

    def objective(trial) -> int:
        """Search of hyperparameters"""
        global BEST_WEIGHTS, BEST_R2

        hidd = trial.suggest_categorical("hidden", [8, 16, 32, 64])
        lr = trial.suggest_float("lr", 0.0001, 0.01)
        ep = trial.suggest_int("ep", 4, 10)
        heads = trial.suggest_categorical("heads", [1, features])
        activ = trial.suggest_categorical("activ", ["leaky_relu", "relu", "elu", "tanh"])

        model = TransformerModel(features, heads, hidd, activation=activ)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # train and test
        model.fit(model, optimizer, ep, train_data)
        r2, mae = model.val(model, test_data)

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
        torch.save(BEST_WEIGHTS, f'best_transformer_weights{ds}.pth')

    return res


if __name__ == "__main__":
    path = '/root/projects/ml-selection/data/processed_data/poly/2_features.csv'
    features = 2
    main(path, features, 1)
