"""
Running hyperparameter selection for all models
"""

from models.hyperparameters_search import (
    gat_hyperparameters_search,
    gcn_hyperparameters_search,
    transformer_hyperparameters_search,
    point_net_hyperparameters_search
)
from models.mathematical_models.turicreate_models import run_math_models

models_in_num = {"0": "GCN", "1": "GAT", "2": "Transformer", "3": "PointNet"}

seebeck_path = "/root/projects/ml-selection/data/raw_data/median_seebeck.json"
poly_dir_path = "/root/projects/ml-selection/data/processed_data/poly/"
poly_path = [
    f"{poly_dir_path}2_features.csv",
    f"{poly_dir_path}3_features.csv",
    f"{poly_dir_path}poly_vector_of_count.csv",
]
poly_just_graph_models = f"{poly_dir_path}0_features.csv"

poly_features = [
    ["poly_elements", "poly_type"],
    ["poly_elements", "poly_vertex", "poly_type"],
    ["poly_elements", "poly_type"],
]
poly_temperature_features = [
    ["poly_elements", "poly_type", "temperature"],
    ["poly_elements", "poly_vertex", "poly_type", "temperature"],
    ["poly_elements", "poly_type", "temperature"],
]
point_features = [3, 4]

total_features = []
total_features.append(poly_features), total_features.append(poly_temperature_features)


def run_net_models() -> list:
    """
    Run hyperparameter search for network models: GCN, GAT, Transformer, PointNet
    """
    total_res_gcn, total_res_gat, total_res_transf, total_res_point_n = [], [], [], []

    # without temperature and without
    for k, features in enumerate(total_features):
        if k == 1:
            temperature = True
        else:
            temperature = False
        for idx, path in enumerate(poly_path):
            print(f"---------START TRAIN NET on descriptor {idx}---------")

            print(f"\n\n---------GCN---------")
            try:
                gcn_res = gcn_hyperparameters_search.main(path, len(features[idx]), idx, temperature)
                total_res_gcn.append(gcn_res)
            except:
                print(f"Failed to start GCN on descriptor: {idx}")

            print(f"\n\n---------GAT---------")
            try:
                gat_res = gat_hyperparameters_search.main(path, len(features[idx]), idx, temperature)
                total_res_gat.append(gat_res)
            except:
                print(f"Failed to start GAT on descriptor: {idx}")

            print(f"\n\n---------Transformer---------")
            try:
                transf_res = transformer_hyperparameters_search.main(
                   path, len(features[idx]), idx, temperature
                )
                total_res_transf.append(transf_res)
            except:
                print(f"Failed to start Transformer on descriptor: {idx}")

    # data consist of structures without size customization
    for k, features in enumerate(total_features):
        if k == 1:
            temperature = True
            feature = 3
        else:
            temperature = False
            feature = 2
        print(f"---------START TRAIN on descriptor {len(poly_path) * 2 + 1 + k}---------")
        print(f"\n\n---------GCN---------")
        try:
            gcn_res = gcn_hyperparameters_search.main(poly_just_graph_models, feature, len(poly_path) + 1 + k, temperature)
            total_res_gcn.append(gcn_res)
        except:
            print(f"Failed to start GCN on descriptor without size customization")

        print(f"\n\n---------GAT---------")
        try:
            gat_res = gat_hyperparameters_search.main(poly_just_graph_models, feature, len(poly_path) + 1 + k, temperature)
            total_res_gat.append(gat_res)
        except:
            print(f"Failed to start GAT on descriptor without size customization")

    print(f"\n\n---------PointNet---------")
    for idx, p_feature in enumerate(point_features):
        print(f"---------START TRAIN on descriptor {idx}---------")
        try:
            pointnet_res = point_net_hyperparameters_search.main(p_feature, idx)
            total_res_point_n.append(pointnet_res)
        except:
            print(f"Failed to start PointNet on descriptor: {idx + 3}D")

    return [total_res_gcn, total_res_gat, total_res_transf, total_res_point_n]


def main():
    """
    Run hyperparameter search for math and network models on PolyDataset and PointCloudDataset
    """
    # format: best_result_r2, mae_for_best_r2, best_model, dataset
    best_res_math = run_math_models(poly_path, seebeck_path, poly_features)
    best_res_temp_math = run_math_models(poly_path, seebeck_path, poly_temperature_features)
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

    if best_res_math[0] >= best_res_temp_math:
        best_res_math_model = best_res_math
    else:
        best_res_math_model= best_res_temp_math
    print("\n------MATH models------")
    print("  R2: ", float(best_res_math_model[0]))
    print("  MAE: ", float(best_res_math_model[1]))
    print("  Model: ", best_res_math_model[2])
    print("  Dataset: ", best_res_math_model[3])


if __name__ == "__main__":
    main()