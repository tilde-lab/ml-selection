"""
Selection of hyperparameters by OPTUNA for ml-models.
"""
import os
import sys
sys.path.append(os.getcwd())

import optuna
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from ml_selection.metrics.compute_metrics import compute_metrics
from skl2onnx.common.data_types import FloatTensorType
import onnx
from skl2onnx import convert_sklearn


BEST_R2, BEST_model = -100, None


def make_study(n_trials, objective_func):
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(), direction="maximize"
    )
    study.optimize(objective_func, n_trials=n_trials)

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
    return res


def run_tune_linear_regression(X_train, y_train, X_test, y_test, n_trials=1, num=250):
    def objective(trial) -> int:
        """Search of hyperparameters"""
        global BEST_R2, BEST_model

        alpha = trial.suggest_float("alpha", 0.0000001, 100.0)

        model = Ridge(alpha)

        # train and test
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2, mae, evs, tur = compute_metrics(
            torch.from_numpy(pred), torch.tensor(y_test)
        )

        print("-------------Ridge-------------")
        print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")

        if r2 > BEST_R2:
            BEST_R2 = r2
            BEST_model = model
            onx = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, 101]))], target_opset=10)
            onnx.save(onx, f"lr_cond_{num}.onnx")

        return r2

    res = make_study(n_trials, objective)
    return res


def run_tune_boosted_trees(X_train, y_train, X_test, y_test, n_trials=1, num=250):
    def objective(trial) -> int:
        """Search of hyperparameters"""
        global BEST_R2, BEST_model

        n_estimators = trial.suggest_int("n_estimators", 2, 300)
        max_depth = trial.suggest_int("max_depth", 2, 300)
        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.05)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 300)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
        max_features = trial.suggest_int("max_features", 1, 300)


        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features
        )

        # train and test
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2, mae, evs, tur = compute_metrics(
            torch.from_numpy(pred), torch.tensor(y_test)
        )

        print("-------------GradientBoostingRegressor-------------")
        print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")

        if r2 > BEST_R2:
            BEST_R2 = r2
            BEST_model = model
            onx = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, 101]))], target_opset=10)
            onnx.save(onx, f"gb_cond_{num}.onnx")

        return r2

    res = make_study(n_trials, objective)
    return res


def run_tune_decision_tree(X_train, y_train, X_test, y_test, n_trials=1, num=250):
    def objective(trial) -> int:
        """Search of hyperparameters"""
        global BEST_R2, BEST_model

        max_depth = trial.suggest_int("max_depth", 3, 300)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 300)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
        min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0001, 1.0)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 3, 200)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_leaf_nodes=max_leaf_nodes
        )

        # train and test
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2, mae, evs, tur = compute_metrics(
            torch.from_numpy(pred), torch.tensor(y_test)
        )

        print("-------------DecisionTreeRegressor-------------")
        print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")

        if r2 > BEST_R2:
            BEST_R2 = r2
            BEST_model = model
            onx = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, 101]))], target_opset=10)
            onnx.save(onx, f"dt_cond_{num}.onnx")

        return r2

    res = make_study(n_trials, objective)

    return res


def run_tune_random_forest(X_train, y_train, X_test, y_test, n_trials=1, num=250):
    def objective(trial) -> int:
        """Search of hyperparameters"""
        global BEST_R2, BEST_model

        max_depth = trial.suggest_int("max_depth", 3, 450)
        n_estimators = trial.suggest_int("n_estimators", 1, 350)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 450)
        min_samples_split = trial.suggest_float("min_samples_split", 0.0, 1.0)
        max_features = trial.suggest_int("max_features", 1, 1000)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features
        )

        # train and test
        model.fit(X_train, y_train)
        print("-------------RandomForestRegressor-------------")

        pred = model.predict(X_test)
        r2, mae, evs, tur = compute_metrics(
            torch.from_numpy(pred), torch.tensor(y_test)
        )

        print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")

        if r2 > BEST_R2:
            BEST_R2 = r2
            BEST_model = model
            onx = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, 101]))], target_opset=10)
            onnx.save(onx, f"rf_cond_{num}.onnx")

        return r2

    res = make_study(n_trials, objective)
    return res

