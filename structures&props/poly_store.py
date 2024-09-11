import yaml


def get_poly_info():
    with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
        yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
        poly_dir_path = yaml_f["polyhedra_path"]

    poly_path = [
        f"{poly_dir_path}101_features.json"
    ]

    poly_features = [
        ["poly_elements", "poly_vertex", "poly_type"],
    ]


    return (
        poly_dir_path,
        poly_path,
        poly_features,
    )
