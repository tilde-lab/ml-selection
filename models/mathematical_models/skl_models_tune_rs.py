"""
Ml-models from sklearn with tuning hyperparameters by RandomizedSearchCV.
"""
import yaml
from typing import Union
from models.hyperparameters_search.randomized_search_ml import (
    run_tune_boosted_trees, run_tune_decision_tree, run_tune_linear_regression,
    run_tune_random_forest)
from skl_models_tune_optuna import load_data, make_descriptors
from data.poly_store import get_poly_info


def main(just_mp: bool = False):
    """Run tune hyp-parameters """
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

    for cnt, f in enumerate([poly_temperature_features]):
        cnt = 1
        print(f"\n\nSTART with descriptor: {f}\n")
        if not just_mp:
            run_ml_models(poly_path, raw_mpds + "median_seebeck.parquet", f, cnt)
        else:
            run_ml_models(poly_path, raw_mpds + "mp_seebeck.parquet", f, cnt)

def run_ml_models(
    poly_paths: list,
    seebeck_path: str,
    f: list,
    is_temp: Union[bool, int],
    n_iter: int = 400,
) -> None:
    for i, poly in enumerate(poly_paths):
        if i != 1:
            continue
        data = load_data(poly, seebeck_path)
        train_x, train_y, test_x, test_y = make_descriptors(data, f, i, is_temp)

        run_tune_linear_regression(train_x, train_y, test_x, test_y, n_iter)
        run_tune_decision_tree(train_x, train_y, test_x, test_y, n_iter)
        run_tune_boosted_trees(train_x, train_y, test_x, test_y, n_iter)
        run_tune_random_forest(train_x, train_y, test_x, test_y, n_iter)


if __name__ == "__main__":
    main()
