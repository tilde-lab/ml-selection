"""Convert skl models to ONNX"""

import pandas as pd
from skl2onnx import to_onnx
import numpy as np
import yaml
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType
import onnx

from typing import Union
from skl_models_tune_optuna import load_data, make_descriptors
from data.poly_store import get_poly_info
from sklearn.ensemble import RandomForestRegressor
import torch
from skl2onnx import convert_sklearn

from metrics.compute_metrics import compute_metrics


def run_ml_model(
    poly_paths: str,
    seebeck_path: str,
    f: list,
    is_temp: Union[bool, int],
) -> tuple:
    """Get dataset, train model, make predict"""
    data = load_data(poly_paths, seebeck_path)
    train_x, train_y, test_x, test_y = make_descriptors(data, f[1], 1, is_temp)

    model = RandomForestRegressor(
        max_depth=230, n_estimators=112,
        min_samples_leaf=2, min_samples_split=0.00010948659349158227,
        max_features=391
    )
    model.fit(train_x.to_numpy().astype(np.float32), train_y.to_numpy().astype(np.float32))

    pred = model.predict(test_x.to_numpy().astype(np.float32))
    r2, mae, evs, tur = compute_metrics(
        torch.from_numpy(pred), torch.tensor(test_y.values)
    )

    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")

    return (model, train_x, train_y, test_x, test_y)


def convert_to_onnx(model: RandomForestRegressor) -> None:
    """
    Convert model into ONNX format. Save in current folder.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model
    """
    initial_type = [('float_input', FloatTensorType([None, 103]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=10)
    onnx_model_path = "random_forest_2.onnx"
    onnx.save(onx, onnx_model_path)


def check_metrics_in_onnx(test_x: pd.DataFrame, test_y: pd.DataFrame):
    """Сheck metrics for ONNX model"""
    sess = rt.InferenceSession("random_forest_2.onnx", providers=rt.get_available_providers())
    pred_ort = sess.run(None, {"float_input": test_x.to_numpy().astype(np.float32)})
    r2, mae, evs, tur = compute_metrics(
        torch.tensor(pred_ort).squeeze(-1).squeeze(-2), torch.tensor(test_y.values)
    )
    print(f"r2: {r2}, mae: {mae}, evs: {evs}, tur: {tur}")


def main():
    """Run """
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
    for i in range(len(poly_path)):
        poly_path[i] = poly_path[i].replace(".json", ".parquet")

    model, train_x, train_y, test_x, test_y = run_ml_model(
        poly_path[1], raw_mpds + "median_seebeck.parquet", poly_temperature_features, 1
    )
    convert_to_onnx(model)
    check_metrics_in_onnx(test_x, test_y)


if __name__ == "__main__":
    main()
