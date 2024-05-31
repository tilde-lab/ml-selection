import yaml

def get_poly_info():
    with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
        yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
        poly_dir_path = yaml_f["polyhedra_path"]

    poly_path = [
        f"{poly_dir_path}2_features.json",
        f"{poly_dir_path}3_features.json",
        f"{poly_dir_path}poly_vector_of_count.json",
    ]
    poly_just_graph_models = f"{poly_dir_path}0_features.json"

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

    return poly_dir_path, poly_path, poly_just_graph_models, poly_features, poly_temperature_features