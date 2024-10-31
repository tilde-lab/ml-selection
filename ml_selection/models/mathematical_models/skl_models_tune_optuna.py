"""
Ml-models from sklearn with tuning hyperparameters by Optuna.
"""
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import polars as pl
import yaml

from ml_selection.models.hyperparameters_search.optuna_ml import \
run_tune_random_forest, run_tune_boosted_trees, run_tune_decision_tree, run_tune_linear_regression


def main(poly_path: str, just_mp: bool = False, phys_prop: str = "Seebeck coefficient"):
    """Run pipeline"""
    with open("ml_selection/configs/config.yaml", "r") as yamlfile:
        yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
        raw_mpds = yaml_f["raw_mpds"]

    # change json on parquet
    poly_path = poly_path.replace(".json", ".parquet")

    if phys_prop == 'Seebeck coefficient':
        if not just_mp:
            run_ml_models(poly_path, raw_mpds + "median_seebeck_mpds.json")
        else:
            # just for seebeck
            run_ml_models(poly_path, raw_mpds + "mp_seebeck.parquet")
    else:
        run_ml_models(poly_path, raw_mpds + "median_conductivity.json")


def load_data(poly_path: str, property_path: str) -> pd.DataFrame:
    """Load data from files"""
    # Crystal in vectors format
    poly = pl.read_parquet(poly_path)
    prop = pl.read_json(property_path)

    # change float to int
    poly = poly.with_columns(pl.col("phase_id").cast(pl.Int64))
    prop = prop.rename({"Phase": "phase_id"})
    data = prop.join(poly, on="phase_id", how="inner")

    return data.to_pandas()


def make_descriptors(data: pd.DataFrame):
    """Create descriptor with same len"""
    def repeat_to_length(lst, length=400):
        while len(lst) < length:
            lst += lst
        return lst[:length]
    
    descriptor = [list(i) for i in data["descriptor"]]
    x = [repeat_to_length(i) for i in descriptor]
    indexes_to_remove = []

    try:
        y = list(data["Seebeck coefficient"])
    except:
        y = list(data["thermal conductivity"])

    train_size = int(0.9 * len(x))
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    return train_x, train_y, test_x, test_y


def run_ml_models(
    poly_path: list,
    phys_prop_path: str,
    n_iter: int = 400,
) -> None:
    data = load_data(poly_path, phys_prop_path)
    train_x, train_y, test_x, test_y = make_descriptors(data)

    run_tune_random_forest(train_x, train_y, test_x, test_y, n_iter)
    run_tune_boosted_trees(train_x, train_y, test_x, test_y, n_iter)
    run_tune_decision_tree(train_x, train_y, test_x, test_y, n_iter)
    run_tune_linear_regression(train_x, train_y, test_x, test_y, n_iter)


if __name__ == "__main__":
    # path to file with next column: phase_id, entry, descriptor
    # main("ml_selection/structures_props/processed_data/descriptor_mpds_seeb_v2.json", False, phys_prop='Seebeck coefficient')
    main("ml_selection/structures_props/processed_data/descriptor_mpds_conductivity_v1.json", False, phys_prop='Conductivity')

