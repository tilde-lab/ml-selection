"""
Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.
"""

import pandas as pd
import polars as pl
import torch
import turicreate as tc
import yaml
from data.poly_store import get_poly_info
from metrics.statistic_metrics import theils_u
from scipy.stats import randint
from sklearn.metrics import explained_variance_score
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
    poly = pl.read_parquet(poly_path)
    seebeck = pl.read_parquet(seebeck_path)

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
    param_distributions = {
        "n_estimators": randint(10, 200),
        "max_depth": randint(1, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 20),
    }

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
    mae_result_r = mae.compute()
    evs = explained_variance_score(
        torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"])
    )
    theils_u_res = theils_u(
        torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"])
    )

    print(
        f"LINEAR REGRESSION MODEL:\nR2: {r2_res_r}, MAE: {mae_result_r}, EVS: {evs}, TUR: {theils_u_res}, min pred: {predictions_linear.min()}, max pred: {predictions_linear.max()}\n\n"
    )
    return [r2_res_r, mae_result_r]


def run_decision_tree(train, test, features):
    # DECISION TREE MODEL
    train_d, test_d = SFrame(train), SFrame(test)

    param_distributions = {
        "max_depth": randint(1, 200),
        "min_child_weight": randint(0.0, 1.0),
        "min_loss_reduction": randint(0.0, 1.0),
    }
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="accuracy",
        n_jobs=-1,
        cv=5,
        verbose=1,
        random_state=42,
    )
    # Обучение модели
    random_search.fit(X_train, y_train)

    # Лучшие гиперпараметры
    best_params = random_search.best_params_
    print("Лучшие гиперпараметры:", best_params)

    # Оценка модели на тестовых данных
    score = random_search.score(X_test, y_test)
    print("Точность на тестовых данных:", score)

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
    evs = explained_variance_score(
        torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"])
    )
    theils_u_res = theils_u(
        torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"])
    )

    print(
        f"DECISION TREE MODEL\nR2: {r2_res_d}, MAE: {mae_result_d}, EVS: {evs}, TUR: {theils_u_res}, min pred: {predictions_decision.min()}, max pred: {predictions_decision.max()}\n\n"
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
    evs = explained_variance_score(
        torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"])
    )
    theils_u_res = theils_u(
        torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"])
    )

    print(
        f"BOOSTED TREES MODEL\nR2: {r2_res_b}, MAE: {mae_result_b}, EVS: {evs}, TUR: {theils_u_res}, min pred: {predictions_boosted.min()}, max pred: {predictions_boosted.max()}\n\n"
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
    evs = explained_variance_score(
        torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"])
    )
    theils_u_res = theils_u(
        torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"])
    )

    print(
        f"RANDOM FOREST MODEL\nR2: {r2_res_rf}, MAE: {mae_result_rf}, EVS: {evs}, TUR: {theils_u_res}, min pred: {predictions_random.min()}, max pred: {predictions_random.max()}\n\n"
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


def main(just_mp: bool = False):
    with open("ml_selection/configs/config.yaml", "r") as yamlfile:
        yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
        raw_mpds = yaml_f["raw_mpds"]

    (
        poly_dir_path,
        poly_path,
        poly_just_graph_models,
        poly_features,
        poly_temperature_features,
    ) = get_poly_info()

    # change json on parquet
    if not just_mp:
        for i in range(len(poly_path)):
            poly_path[i] = poly_path[i].replace(".json", ".parquet")
    else:
        for i in range(len(poly_path)):
            poly_path[i] = poly_path[i].replace(".json", "_mp.parquet")

    for f in [poly_features, poly_temperature_features]:
        if not just_mp:
            run_math_models(poly_path, raw_mpds + "median_seebeck.parquet", f)
        else:
            run_math_models(poly_path, raw_mpds + "mp_seebeck.parquet", f)


if __name__ == "__main__":
    main(just_mp=False)
