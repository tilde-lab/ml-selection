"""
Ml-models
"""

import numpy as np
import polars as pl

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from data.poly_store import get_poly_info
import yaml
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


def main(just_mp: bool = False):
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

    # TODO: add poly_temperature_features
    for f in [poly_features]:
        print(f'\n\nSTART with descriptor: {f}\n')
        if not just_mp:
            run_ml_models(poly_path, raw_mpds + "median_seebeck.parquet", f)
        else:
            run_ml_models(poly_path, raw_mpds + "mp_seebeck.parquet", f)


def split_data_for_models(poly_path: str, seebeck_path: str) -> pd.DataFrame:
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


def run_ml_models(poly_paths: list, seebeck_path: str, f: list, n_iter: int = 1) -> None:
    for i, poly in enumerate(poly_paths):
        print(f'File: {poly}')
        train, test = split_data_for_models(poly, seebeck_path)

        train_y, test_y = train['Seebeck coefficient'], test['Seebeck coefficient']

        train_x = train.drop(columns=['Seebeck coefficient', 'Formula', 'phase_id', 'temperature'])
        test_x = test.drop(columns=['Seebeck coefficient', 'Formula'])

        poly_elements_df = pd.DataFrame(train_x['poly_elements'].tolist())
        poly_type_df = pd.DataFrame(train_x['poly_type'].tolist())
        if len(f[i]) == 2:
            train_x = pd.concat([poly_elements_df, poly_type_df], axis=1)
        else:
            poly_v = pd.DataFrame(train_x['poly_vertex'].tolist())
            train_x = pd.concat([poly_elements_df, poly_type_df, poly_v], axis=1)

        poly_elements_df = pd.DataFrame(test_x['poly_elements'].tolist())
        poly_type_df = pd.DataFrame(test_x['poly_type'].tolist())
        if len(f[i]) == 2:
            test_x = pd.concat([poly_elements_df, poly_type_df], axis=1)
        else:
            poly_v = pd.DataFrame(test_x['poly_vertex'].tolist())
            test_x = pd.concat([poly_elements_df, poly_type_df, poly_v], axis=1)

        run_linear_regression(train_x, train_y, test_x, test_y, n_iter)
        run_decision_tree(train_x, train_y, test_x, test_y, n_iter)
        run_boosted_trees(train_x, train_y, test_x, test_y, n_iter)
        run_random_forest(train_x, train_y, test_x, test_y, n_iter)


def run_linear_regression(X_train, y_train, X_test, y_test, n_iter):
    ridge = Ridge()
    ridge_param_distributions = {
        'alpha': np.random.uniform(0.00001, 10.0, size=10000)
    }
    ridge_search = RandomizedSearchCV(
        estimator=ridge,
        param_distributions=ridge_param_distributions,
        n_iter=n_iter,
        scoring='r2',
        cv=5,
        verbose=1,
        random_state=42
    )
    ridge_search.fit(X_train, y_train)
    print("Best parms for Ridge:", ridge_search.best_params_)
    score = ridge_search.score(X_test, y_test)
    print("R2:", score)

    return ridge_search.best_params_


def run_boosted_trees(X_train, y_train, X_test, y_test, n_iter):
    gbm = GradientBoostingRegressor()
    gbm_param_distributions = {
        'n_estimators': randint(1, 200),
        'learning_rate': np.random.uniform(0.000001, 0.5, size=100000),
        'max_depth': randint(3, 100)
    }

    gbm_search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=gbm_param_distributions,
        n_iter=n_iter,
        scoring='r2',
        cv=5,
        verbose=1,
        random_state=42
    )
    gbm_search.fit(X_train, y_train)
    print("Best parms for Gradient Boosting:", gbm_search.best_params_)
    score = gbm_search.score(X_test, y_test)
    print("R2:", score)

    return gbm_search.best_params_


def run_decision_tree(X_train, y_train, X_test, y_test, n_iter):
    decision_tree = DecisionTreeRegressor()
    param_distributions = {
        'max_depth': randint(5, 150),
        'min_samples_split': randint(1, 100),
        'min_samples_leaf': randint(1, 100)
    }

    random_search = RandomizedSearchCV(
        estimator=decision_tree,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='r2',
        cv=5,
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best parms for DecisionTree:", best_params)
    score = random_search.score(X_test, y_test)
    print("R2:", score)

    return best_params


def run_random_forest(X_train, y_train, X_test, y_test, n_iter):
    rf = RandomForestRegressor()
    rf_param_distributions = {
        'n_estimators': randint(1, 100),
        'max_depth': randint(1, 100),
        'min_samples_split': randint(2, 100)
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_distributions,
        n_iter=n_iter,
        scoring='r2',
        cv=5,
        verbose=1,
        random_state=42
    )
    rf_search.fit(X_train, y_train)

    best_params = rf_search.best_params_
    print("Best parms for Random Forest:", best_params)
    score = rf_search.score(X_test, y_test)
    print("R2:", score)

    return best_params


if __name__ == "__main__":
    poly_paths, seebeck_path, features = main()
