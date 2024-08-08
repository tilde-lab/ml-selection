"""
Ml-models from sklearn with tuning hyperparameters by Optuna.
"""

from typing import Union

import pandas as pd
import polars as pl
import yaml

from data.poly_store import get_poly_info
from models.hyperparameters_search.optuna_ml import run_tune_random_forest


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
            run_ml_models(poly_path, raw_mpds + "median_conductivity.json", f, cnt)
        else:
            # just for seebeck
            run_ml_models(poly_path, raw_mpds + "mp_seebeck.parquet", f, cnt)


def load_data(poly_path: str, property_path: str) -> pd.DataFrame:
    """Load data from files"""
    # Crystal in vectors format
    poly = pl.read_parquet(poly_path)
    prop = pl.read_json(property_path).rename({'Phase': 'phase_id'})

    # change float to int
    poly = poly.with_columns(pl.col("phase_id").cast(pl.Int64))
    data = prop.join(poly, on="phase_id", how="inner")

    return data.to_pandas()


def make_descriptors(data: pd.DataFrame, f: list, i: int, is_temp: bool):
    """Create 6 different type of descriptor"""
    poly_elements_df = pd.DataFrame(data["poly_elements"].tolist())
    # there is no need to have same dim - take a number instead of an array
    poly_type_df = pd.DataFrame([[row[0]] for row in data["poly_type"].values.tolist()])
    y = data["thermal conductivity"]

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
    phys_prop_path: str,
    f: list,
    is_temp: Union[bool, int],
    n_iter: int = 2000,
) -> None:
    for i, poly in enumerate(poly_paths):
        data = load_data(poly, phys_prop_path)
        train_x, train_y, test_x, test_y = make_descriptors(data, f, i, is_temp)

        run_tune_random_forest(train_x, train_y, test_x, test_y, n_iter, num=str(i) + str(len(f[0])-1))


if __name__ == "__main__":
    main()
