"""
Ml-models from sklearn with tuning hyperparameters by Optuna.
"""

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
        poly_features,
    ) = get_poly_info()

    # change json on parquet
    if not just_mp:
        for i in range(len(poly_path)):
            poly_path[i] = poly_path[i].replace(".json", ".parquet")
    else:
        for i in range(len(poly_path)):
            poly_path[i] = poly_path[i].replace(".json", "_mp.parquet")

    if not just_mp:
        run_ml_models(poly_path, raw_mpds + "median_seebeck.json")
    else:
        # just for seebeck
        run_ml_models(poly_path, raw_mpds + "mp_seebeck.parquet")


def load_data(poly_path: str, property_path: str) -> pd.DataFrame:
    """Load data from files"""
    # Crystal in vectors format
    poly = pl.read_parquet(poly_path)
    prop = pl.read_json(property_path)

    # change float to int
    poly = poly.with_columns(pl.col("phase_id").cast(pl.Int64))
    data = prop.join(poly, on="phase_id", how="inner")

    return data.to_pandas()


def make_descriptors(data: pd.DataFrame):
    """Create 6 different type of descriptor"""
    poly_elements_df = pd.DataFrame(data["poly_elements"].tolist())
    # there is no need to have same dim - take a number instead of an array
    poly_type_df = pd.DataFrame(data["poly_type"].values.tolist())
    y = data["Seebeck coefficient"]

    x = pd.concat([poly_elements_df, poly_type_df], axis=1)
    # take a number instead of an array too

    train_size = int(0.9 * len(data))
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    return train_x, train_y, test_x, test_y


def run_ml_models(
    poly_paths: list,
    phys_prop_path: str,
    n_iter: int = 3000,
) -> None:
    for i, poly in enumerate(poly_paths):
        data = load_data(poly, phys_prop_path)
        train_x, train_y, test_x, test_y = make_descriptors(data)

        run_tune_random_forest(train_x, train_y, test_x, test_y, n_iter, num=str(101))


if __name__ == "__main__":
    main(False)
