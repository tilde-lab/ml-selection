"""
Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.
"""

import pandas as pd
import polars as pl
import torch
import turicreate as tc
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from turicreate import SFrame

mae = MeanAbsoluteError()
metric = R2Score()
mape = MeanAbsolutePercentageError()

models = {
    "0": "LINEAR REGRESSION",
    "1": "DECISION TREE MODEL",
    "2": "BOOSTED TREES MODEL",
    "3": "RANDOM FOREST MODEL",
}


def split_data_for_turi_models(poly_path: str, seebeck_path: str) -> pd.DataFrame:
    # Crystal in vectors format
    poly = pl.read_json(poly_path)
    seebeck = pl.read_json(seebeck_path)

    # change float to int
    poly = poly.with_columns(pl.col("phase_id").cast(pl.Int64))
    data = seebeck.join(poly, on="phase_id", how="inner")

    train_size = int(0.9 * len(data))

    train = data[:train_size]
    test = data[train_size:]

    return (train.to_pandas(), test.to_pandas())


def run_linear_regression(train, test, features):
    # LINEAR REGRESSION MODEL
    train_r, test_r = SFrame(train), SFrame(test)
    model_linear = tc.linear_regression.create(
        train_r, target="Seebeck coefficient", features=features, validation_set=test_r
    )
    predictions_linear = model_linear.predict(test_r)

    metric.update(
        torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"])
    )
    r2_res_r = metric.compute()
    mae.update(
        torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"])
    )
    mape.update(
        torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"])
    )
    mape_res = mape.compute()
    mae_result_r = mae.compute()
    print(
        f"LINEAR REGRESSION MODEL:\nR2: {r2_res_r}, MAE: {mae_result_r}, MAPE: {mape_res}, min pred: {predictions_linear.min()}, max pred: {predictions_linear.max()}\n\n"
    )
    return [r2_res_r, mae_result_r]


def run_decision_tree(train, test, features):
    # DECISION TREE MODEL
    train_d, test_d = SFrame(train), SFrame(test)
    model_decision = tc.decision_tree_regression.create(
        train_d, target="Seebeck coefficient", features=features, validation_set=test_d
    )
    predictions_decision = model_decision.predict(test_d)

    metric.update(
        torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"])
    )
    r2_res_d = metric.compute()
    mae.update(
        torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"])
    )
    mae_result_d = mae.compute()
    mape.update(
        torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"])
    )
    mape_res = mape.compute()
    print(
        f"DECISION TREE MODEL\nR2: {r2_res_d}, MAE: {mae_result_d}, MAPE: {mape_res}, min pred: {predictions_decision.min()}, max pred: {predictions_decision.max()}\n\n"
    )
    return [r2_res_d, mae_result_d]


def run_boosted_trees(train, test, features):
    # BOOSTED TREES MODEL
    train_b, test_b = SFrame(train), SFrame(test)
    model_boosted = tc.boosted_trees_regression.create(
        train_b, target="Seebeck coefficient", features=features, validation_set=test_b
    )
    predictions_boosted = model_boosted.predict(test_b)
    metric.update(
        torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"])
    )
    r2_res_b = metric.compute()
    mae.update(
        torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"])
    )
    mae_result_b = mae.compute()
    mape.update(
        torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"])
    )
    mape_res = mape.compute()

    print(
        f"BOOSTED TREES MODEL\nR2: {r2_res_b}, MAE: {mae_result_b}, MAPE: {mape_res}, min pred: {predictions_boosted.min()}, max pred: {predictions_boosted.max()}\n\n"
    )
    return [r2_res_b, mae_result_b]


def run_random_forest(train, test, features):
    # RANDOM FOREST MODEL
    train_r, test_r = SFrame(train), SFrame(test)
    model_random = tc.random_forest_regression.create(
        train_r, target="Seebeck coefficient", features=features, validation_set=test_r
    )
    predictions_random = model_random.predict(test_r)
    metric.update(
        torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"])
    )
    r2_res_rf = metric.compute()
    mae.update(
        torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"])
    )
    mae_result_rf = mae.compute()
    mape.update(
        torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"])
    )
    mape_res = mape.compute()
    print(
        f"RANDOM FOREST MODEL\nR2: {r2_res_rf}, MAE: {mae_result_rf}, MAPE: {mape_res} min pred: {predictions_random.min()}, max pred: {predictions_random.max()}\n\n"
    )
    return [r2_res_rf, mae_result_rf]


def run_math_models(poly_paths: list, seebeck_path: str, features: list) -> None:
    result = []

    for i, poly in enumerate(poly_paths):
        metrics = []
        train, test = split_data_for_turi_models(poly, seebeck_path)
        metrics.append(run_linear_regression(train, test, features[i]))
        metrics.append(run_decision_tree(train, test, features[i]))
        metrics.append(run_boosted_trees(train, test, features[i]))
        metrics.append(run_random_forest(train, test, features[i]))
        result.append(metrics)

    best_result_r2 = -1
    mae_for_best_r2 = None
    best_model = None
    dataset = 1

    for idx, data in enumerate(result):
        r2 = [i[0] for i in data]
        mae = [i[1] for i in data]
        best_for_that_data = max(r2)
        print(
            f"Best result in DATASET {idx + 1}: {max(r2)}, model: {models[str(r2.index(max(r2)))]}"
        )

        if best_for_that_data > best_result_r2:
            best_result_r2 = best_for_that_data
            best_model = models[str(r2.index(max(r2)))]
            dataset = idx + 1
            mae_for_best_r2 = mae[r2.index(max(r2))]

    print(
        f"\n\nBest result from all experiments: {best_result_r2}, model: {best_model}, dataset: {dataset}"
    )
    return (best_result_r2, mae_for_best_r2, best_model, dataset)


if __name__ == "__main__":
    seebeck_path = "/data/raw_mpds/median_seebeck.json"
    poly_path = [
        "/root/projects/ml-selection/data/processed_data/poly/2_features.json",
        "/root/projects/ml-selection/data/processed_data/poly/3_features.json",
        "/root/projects/ml-selection/data/processed_data/poly/poly_vector_of_count.json",
    ]

    features = [
        ["poly_elements", "poly_type"],
        ["poly_elements", "poly_vertex", "poly_type"],
        ["poly_elements", "poly_type"],
    ]
    poly_temperature_features = [
        ["poly_elements", "poly_type", "temperature"],
        ["poly_elements", "poly_vertex", "poly_type", "temperature"],
        ["poly_elements", "poly_type", "temperature"],
    ]

    run_math_models(poly_path, seebeck_path, features)
