"""Convert skl models to ONNX"""

from typing import Union

import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd
import torch
import yaml
from data.poly_store import get_poly_info
from metrics.compute_metrics import compute_metrics
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl_models_tune_optuna import load_data, make_descriptors
from sklearn.ensemble import RandomForestRegressor


def run_ml_model(
    poly_paths: str,
    seebeck_path: str,
    f: list,
    is_temp: Union[bool, int],
) -> tuple:
    """Get dataset, train model, make predict"""
    data = load_data(poly_paths, seebeck_path)
    train_x, train_y, test_x, test_y = make_descriptors(data, f[1], 1, is_temp)
    r2 = 0

    model = RandomForestRegressor(
        n_estimators=259,
        max_depth=184,
        min_samples_leaf=2,
        min_samples_split=0.0008226399734106349,
        max_features=285,
    )
    model.fit(
        train_x.to_numpy().astype(np.float32), train_y.to_numpy().astype(np.float32)
    )

    pred = model.predict(test_x.to_numpy().astype(np.float32))
    r2, mae, evs, tur = compute_metrics(
        torch.from_numpy(pred), torch.tensor(test_y.values)
    )

    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")

    return (model, train_x, train_y, test_x, test_y)


def convert_to_onnx(model: RandomForestRegressor, num: str) -> None:
    """
    Convert model into ONNX format. Save in current folder.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model
    """
    initial_type = [("input", FloatTensorType([None, 103]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=10)
    onnx_model_path = f"random_forest_conductivity_{num}.onnx"
    onnx.save(onx, onnx_model_path)


def check_metrics_in_onnx(test_x: pd.DataFrame, test_y: pd.DataFrame):
    """Сheck metrics for ONNX model"""
    sess = rt.InferenceSession(
        "random_forest.onnx", providers=rt.get_available_providers()
    )
    pred_ort = sess.run(None, {"input": test_x.to_numpy().astype(np.float32)})
    r2, mae, evs, tur = compute_metrics(
        torch.tensor(pred_ort).squeeze(-1).squeeze(-2), torch.tensor(test_y.values)
    )
    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")


def main():
    """Run"""
    with open("ml_selectionn/configs/config.yaml", "r") as yamlfile:
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
    for i in range(len(poly_path)):
        poly_path[i] = poly_path[i].replace(".json", ".parquet")

    model, train_x, train_y, test_x, test_y = run_ml_model(
        poly_path[1], raw_mpds + "median_seebeck.parquet", poly_temperature_features, 1
    )
    convert_to_onnx(model)
    check_metrics_in_onnx(test_x, test_y)


if __name__ == "__main__":
    main()
