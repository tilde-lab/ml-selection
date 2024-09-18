"""
Running hyperparameter selection for all models
"""
import yaml
from data.poly_store import get_poly_info
from models.hyperparameters_search import (gat_hyperparameters_search,
                                           gcn_hyperparameters_search,
                                           point_net_hyperparameters_search,
                                           transformer_hyperparameters_search)

from models.neural_network_models.GCN import gcn_regression_model
from models.neural_network_models.GAT import gat_regression_model
from models.neural_network_models.PointNet import pointnet_model
from models.neural_network_models.transformer import transformer_reg

CONF = "/root/projects/ml-selection/configs/config.yaml"

with open(CONF, "r") as yamlfile:
    yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
    raw_mpds = yaml_f["raw_mpds"]

models_in_num = {"0": "GCN", "1": "GAT", "2": "Transformer", "3": "PointNet"}
seebeck_path = f"{raw_mpds}median_seebeck.json"
(
        poly_dir_path,
        poly_path,
        poly_just_graph_models,
        poly_features,
        poly_temperature_features,
    ) = get_poly_info()
point_features = [3, 4]

total_features = []
total_features.append(poly_features), total_features.append(poly_temperature_features)


def run_net_models(epoch: int, name_to_save_w: str, just_mp: bool = False):
    print('START TRANSFORMER')
    transformer_reg.main(epoch=epoch, name_to_save=name_to_save_w, just_mp=just_mp)
    print('START GCN')
    gcn_regression_model.main(epoch=epoch, name_to_save=name_to_save_w, just_mp=just_mp)
    print('START GAT')
    gat_regression_model.main(epoch=epoch, name_to_save=name_to_save_w, just_mp=just_mp)
    print('START PONIT NET')
    pointnet_model.main(epoch=epoch, name_to_save=name_to_save_w, just_mp=just_mp)


def run_hypp_search_net_models(with_except=True) -> list:
    """
    Run hyperparameter search for network models: GCN, GAT, Transformer, PointNet
    """
    total_res_gcn, total_res_gat, total_res_transf, total_res_point_n = [], [], [], []

    if with_except:
        # without temperature
        for k, features in enumerate(total_features):
            if k == 1:
                temperature = True
            else:
                temperature = False
            for idx, path in enumerate(poly_path):
                print(f"---------START TRAIN NET on descriptor {idx}---------")

                print(f"\n\n---------GCN---------")
                try:
                    gcn_res = gcn_hyperparameters_search.main(
                        path, len(features[idx]), idx, temperature
                    )
                    total_res_gcn.append(gcn_res)
                except:
                    print(f"Failed to start GCN on descriptor: {idx}")

                print(f"\n\n---------GAT---------")
                try:
                    gat_res = gat_hyperparameters_search.main(
                        path, len(features[idx]), idx, temperature
                    )
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
            print(
                f"---------START TRAIN on descriptor {len(poly_path) * 2 + 1 + k}---------"
            )
            print(f"\n\n---------GCN---------")
            try:
                gcn_res = gcn_hyperparameters_search.main(
                    poly_just_graph_models, feature, len(poly_path) + 1 + k, temperature
                )
                total_res_gcn.append(gcn_res)
            except:
                print(f"Failed to start GCN on descriptor without size customization")

            print(f"\n\n---------GAT---------")
            try:
                gat_res = gat_hyperparameters_search.main(
                    poly_just_graph_models, feature, len(poly_path) + 1 + k, temperature
                )
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
    else:
        # without temperature
        for k, features in enumerate(total_features):
            if k == 1:
                temperature = True
            else:
                temperature = False
            for idx, path in enumerate(poly_path):
                print(f"---------START TRAIN NET on descriptor {idx}---------")

                print(f"\n\n---------GCN---------")
                gcn_res = gcn_hyperparameters_search.main(path, len(features[idx]), idx, temperature)
                total_res_gcn.append(gcn_res)

                print(f"\n\n---------GAT---------")
                gat_res = gat_hyperparameters_search.main(path, len(features[idx]), idx, temperature)
                total_res_gat.append(gat_res)

                print(f"\n\n---------Transformer---------")
                transf_res = transformer_hyperparameters_search.main(
                    path, len(features[idx]), idx, temperature
                )
                total_res_transf.append(transf_res)

        # data consist of structures without size customization
        for k, features in enumerate(total_features):
            if k == 1:
                temperature = True
                feature = 3
            else:
                temperature = False
                feature = 2
            print(
                f"---------START TRAIN on descriptor {len(poly_path) * 2 + 1 + k}---------"
            )
            print(f"\n\n---------GCN---------")
            gcn_res = gcn_hyperparameters_search.main(
                poly_just_graph_models, feature, len(poly_path) + 1 + k, temperature
            )
            total_res_gcn.append(gcn_res)

            print(f"\n\n---------GAT---------")
            gat_res = gat_hyperparameters_search.main(
                poly_just_graph_models, feature, len(poly_path) + 1 + k, temperature
            )
            total_res_gat.append(gat_res)

        print(f"\n\n---------PointNet---------")
        for idx, p_feature in enumerate(point_features):
            print(f"---------START TRAIN on descriptor {idx}---------")
            pointnet_res = point_net_hyperparameters_search.main(p_feature, idx)
            total_res_point_n.append(pointnet_res)

    return [total_res_gcn, total_res_gat, total_res_transf, total_res_point_n]


def main_hypp():
    """
    Run hyperparameter search for math and network models on PolyDataset and PointCloudDataset
    """
    total_res_nets = run_hypp_search_net_models(with_except=False)

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


if __name__ == "__main__":
    run_net_models(epoch=10, name_to_save_w='04_06', just_mp=True)
