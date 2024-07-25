import numpy as np
import torch
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from models.mathematical_models.sklearn_models import compute_metrics


def run_tune_linear_regression(X_train, y_train, X_test, y_test, n_iter):
    ridge = Ridge()
    ridge_param_distributions = {
        "alpha": np.random.uniform(0.0000001, 100.0, size=100000)
    }
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


def run_tune_boosted_trees(X_train, y_train, X_test, y_test, n_iter):
    gbm = GradientBoostingRegressor()
    gbm_param_distributions = {
        "n_estimators": randint(1, 100),
        "learning_rate": np.random.uniform(0.000001, 0.5, size=100000),
        "max_depth": randint(3, 100),
        "min_samples_leaf": randint(3, 100),
        "min_samples_split": randint(3, 100),
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


def run_tune_decision_tree(X_train, y_train, X_test, y_test, n_iter):
    decision_tree = DecisionTreeRegressor()
    param_distributions = {
        "max_depth": randint(1, 200),
        "min_samples_split": randint(3, 100),
        "min_samples_leaf": randint(3, 100),
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


def run_tune_random_forest(X_train, y_train, X_test, y_test, n_iter):
    rf = RandomForestRegressor()
    rf_param_distributions = {
        "n_estimators": randint(1, 100),
        "max_depth": randint(1, 100),
        "min_samples_split": randint(3, 100),
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
