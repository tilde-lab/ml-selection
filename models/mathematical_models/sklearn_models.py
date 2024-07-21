"""
Ml-models from sklearn with tuning hyperparameters by RandomizedSearchCV.
"""

import torch
import numpy as np
import polars as pl
from torcheval.metrics import R2Score
from typing import Union

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from data.poly_store import get_poly_info
import yaml
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

from torchmetrics import MeanAbsoluteError
from data_massage.metrics.statistic_metrics import theils_u
from sklearn.metrics import explained_variance_score


def main(just_mp: bool = False):
    """Run pipeline"""
    with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
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

    for cnt, f in enumerate([poly_features, poly_temperature_features]):
        print(f"\n\nSTART with descriptor: {f}\n")
        if not just_mp:
            run_ml_models(poly_path, raw_mpds + "median_seebeck.parquet", f, cnt)
        else:
            run_ml_models(poly_path, raw_mpds + "mp_seebeck.parquet", f, cnt)


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple:
    """Compute R2, MAE, EVS, Theil's U metrics"""
    mae, r2 = MeanAbsoluteError(), R2Score()

    mae.update(y_pred, y_true)
    r2.update(y_pred, y_true)

    mae_result = mae.compute()
    r2_res = r2.compute()
    evs = explained_variance_score(y_pred, y_true)
    theils_u_res = theils_u(y_pred, y_true)

    return (r2_res, mae_result, evs, theils_u_res)


def load_data(poly_path: str, seebeck_path: str) -> pd.DataFrame:
    """Load data from '.parquet' files"""
    # Crystal in vectors format
    poly = pl.read_parquet(poly_path)
    seebeck = pl.read_parquet(seebeck_path)

    # change float to int
    poly = poly.with_columns(pl.col("phase_id").cast(pl.Int64))
    data = seebeck.join(poly, on="phase_id", how="inner")

    return data.to_pandas()


def make_descriptors(data: pd.DataFrame, f: list, i: int, is_temp: bool):
    """Create 6 different type of descriptor"""
    poly_elements_df = pd.DataFrame(data["poly_elements"].tolist())
    # there is no need to have same dim - take a number instead of an array
    poly_type_df = pd.DataFrame([[row[0]] for row in data["poly_type"].values.tolist()])
    y = data["Seebeck coefficient"]

    if is_temp:
        temperature = data["temperature"]
        if len(f[i]) == 3:
            x = pd.concat([poly_elements_df, poly_type_df, temperature], axis=1)
        else:
            # take a number instead of an array too
            poly_v = pd.DataFrame(
                [[row[0]] for row in data["poly_vertex"].values.tolist()]
            )
            x = pd.concat([poly_elements_df, poly_type_df, poly_v, temperature], axis=1)
    else:
        if len(f[i]) == 2:
            x = pd.concat([poly_elements_df, poly_type_df], axis=1)
        else:
            # take a number instead of an array too
            poly_v = pd.DataFrame(
                [[row[0]] for row in data["poly_vertex"].values.tolist()]
            )
            x = pd.concat([poly_elements_df, poly_type_df, poly_v], axis=1)

    train_size = int(0.9 * len(data))
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    return train_x, train_y, test_x, test_y


def run_ml_models(
    poly_paths: list,
    seebeck_path: str,
    f: list,
    is_temp: Union[bool, int],
    n_iter: int = 100,
) -> None:
    for i, poly in enumerate(poly_paths):
        print(f"File: {poly}")
        data = load_data(poly, seebeck_path)
        train_x, train_y, test_x, test_y = make_descriptors(data, f, i, is_temp)

        run_linear_regression(train_x, train_y, test_x, test_y, n_iter)
        run_decision_tree(train_x, train_y, test_x, test_y, n_iter)
        run_boosted_trees(train_x, train_y, test_x, test_y, n_iter)
        run_random_forest(train_x, train_y, test_x, test_y, n_iter)


def run_linear_regression(X_train, y_train, X_test, y_test, n_iter):
    ridge = Ridge()
    ridge_param_distributions = {"alpha": np.random.uniform(0.0000001, 100.0, size=100000)}
    ridge_search = RandomizedSearchCV(
        estimator=ridge,
        param_distributions=ridge_param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=5,
        verbose=1,
        random_state=42,
    )
    ridge_search.fit(X_train, y_train)
    pred = ridge_search.predict(X_test)
    r2, mae, evs, tur = compute_metrics(
        torch.from_numpy(pred), torch.tensor(y_test.values)
    )

    print("Best parms for Ridge:", ridge_search.best_params_)
    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")


def run_boosted_trees(X_train, y_train, X_test, y_test, n_iter):
    gbm = GradientBoostingRegressor()
    gbm_param_distributions = {
        "n_estimators": randint(1, 100),
        "learning_rate": np.random.uniform(0.000001, 0.5, size=100000),
        "max_depth": randint(3, 100),
        "min_samples_leaf": randint(1, 100),
        "min_samples_split": randint(1, 100),
    }

    gbm_search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=gbm_param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=5,
        verbose=1,
        random_state=42,
    )
    gbm_search.fit(X_train, y_train)
    pred = gbm_search.predict(X_test)
    r2, mae, evs, tur = compute_metrics(
        torch.from_numpy(pred), torch.tensor(y_test.values)
    )

    print("Best parms for Gradient Boosting:", gbm_search.best_params_)
    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")


def run_decision_tree(X_train, y_train, X_test, y_test, n_iter):
    decision_tree = DecisionTreeRegressor()
    param_distributions = {
        "max_depth": randint(1, 200),
        "min_samples_split": randint(1, 100),
        "min_samples_leaf": randint(1, 100),
    }

    random_search = RandomizedSearchCV(
        estimator=decision_tree,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=5,
        verbose=1,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    pred = random_search.predict(X_test)
    r2, mae, evs, tur = compute_metrics(
        torch.from_numpy(pred), torch.tensor(y_test.values)
    )

    print("Best parms for DecisionTree:", random_search.best_params_)
    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")


def run_random_forest(X_train, y_train, X_test, y_test, n_iter):
    rf = RandomForestRegressor()
    rf_param_distributions = {
        "n_estimators": randint(1, 100),
        "max_depth": randint(1, 100),
        "min_samples_split": randint(1, 100),
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=5,
        verbose=1,
        random_state=42,
    )
    rf_search.fit(X_train, y_train)

    pred = rf_search.predict(X_test)
    r2, mae, evs, tur = compute_metrics(
        torch.from_numpy(pred), torch.tensor(y_test.values)
    )

    print("Best parms for Random Forest:", rf_search.best_params_)
    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")


if __name__ == "__main__":
    main()
