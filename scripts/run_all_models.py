"""
Running hyperparameter selection for all models
"""

from models.hyperparameters_search import (
    gat_hyperparameters_search,
    gcn_hyperparameters_search,
    transformer_hyperparameters_search,
)
from models.mathematical_models.turicreate_models import run_math_models

models_in_num = {"0": "GCN", "1": "GAT", "2": "Transformer"}

seebeck_path = "/root/projects/ml-selection/data/raw_data/median_seebeck.json"
dir_path = "/root/projects/ml-selection/data/processed_data/poly/"
poly_path = [
    f"{dir_path}2_features.csv",
    f"{dir_path}3_features.csv",
    f"{dir_path}poly_vector_of_count.csv",
]
poly_just_graph_models = f"{dir_path}0_features.csv"

features = [
    ["poly_elements", "poly_type"],
    ["poly_elements", "poly_vertex", "poly_type"],
    ["poly_elements", "poly_type"],
]


def run_net_models():
    total_res_gcn, total_res_gat, total_res_transf = [], [], []

    for idx, path in enumerate(poly_path):
        print(f"---------START TRAIN NET on data {idx}---------")

        print(f"\n\n---------GCN---------")
        try:
            gcn_res = gcn_hyperparameters_search.main(path, len(features[idx]), idx)
            total_res_gcn.append(gcn_res)
        except:
            print(f"Failed to start GCN on Dataset: {idx}")

        print(f"\n\n---------GAT---------")
        try:
            gat_res = gat_hyperparameters_search.main(path, len(features[idx]), idx)
            total_res_gat.append(gat_res)
        except:
            print(f"Failed to start GAT on Dataset: {idx}")

        print(f"\n\n---------Transformer---------")

        try:
            transf_res = transformer_hyperparameters_search.main(
                path, len(features[idx]), idx
            )
            total_res_transf.append(transf_res)
        except:
            print(f"Failed to start Transformer on Dataset: {idx}")

    print(f"---------START TRAIN on data {len(poly_path) + 1}---------")

    print(f"\n\n---------GCN---------")
    gcn_res = gcn_hyperparameters_search.main(poly_just_graph_models, 2, len(poly_path) + 1)
    print(f"\n\n---------GAT---------")
    gat_res = gat_hyperparameters_search.main(poly_just_graph_models, 2, len(poly_path) + 1)

    total_res_gcn.append(gcn_res), total_res_gat.append(gat_res)

    return [total_res_gcn, total_res_gat, total_res_transf]


def main():
    # format: best_result_r2, mae_for_best_r2, best_model, dataset
    best_res_math = run_math_models(poly_path, seebeck_path, features)
    total_res_nets = run_net_models()

    print("\n\n------TOTAL BEST RESULT------")
    for i, model_res in enumerate(total_res_nets):
        best_r2 = -100
        best_data = None
        type_of_dataset = None

        print(f"\n------{models_in_num[str(i)]}------")
        for idx, res in enumerate(model_res):
            if res[0].values[0] > best_r2:
                best_r2 = res[0].values[0]
                best_data = res[0]
                type_of_dataset = idx

        print("  R2: ", best_r2)
        print("  Params: ")
        for key, value in best_data.params.items():
            print(f"    {key}: {value}")
        print("  Dataset: ", type_of_dataset)

    print("\n------MATH models------")
    print("  R2: ", float(best_res_math[0]))
    print("  MAE: ", float(best_res_math[1]))
    print("  Model: ", best_res_math[2])
    print("  Dataset: ", best_res_math[3])


if __name__ == "__main__":
    main()
