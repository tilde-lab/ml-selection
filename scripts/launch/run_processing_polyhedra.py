"""
Run add polyhedra to structures by entry, make descriptor
"""

from data_massage.data_handler import DataHandler
import yaml


hand = DataHandler(True)

CONF = "/root/projects/ml-selection/configs/config.yaml"

with open(CONF, "r") as yamlfile:
    conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    raw_data = conf["raw_mpds"]
    processed_data = conf["polyhedra_path"]


def combine_structure_and_poly(just_mp: bool) -> None:
    """
    Combine structures and polyhedra by entry
    """
    if just_mp:
        dfrm = hand.add_polyhedra(raw_data + "mp_structures.json")
        dfrm.write_json(raw_data + "large_poly_mp.json")
    else:
        dfrm = hand.add_polyhedra(raw_data + "rep_structures.json")
        dfrm.write_json(raw_data + "large_poly.json")


def make_poly_descriptor(
    features: int = 2, file_name: str = "test", is_one_hot: bool = False
) -> None:
    """
    Run creating polyhedra descriptor

    Parameters
    ----------
    features: int
        Number or type of expected features:
        2 -> 2 features
        3 -> 3 features
        0 -> structure without size customization (all graphs consist of different number of nodes)
    file_name : str, optional
        Name for result file
    is_one_hot : bool, optional
        If True -> create vector of elements from polyhedra in one-hot format
    """
    descriptor = hand.process_polyhedra(
        raw_data + "large_poly.json", features=features, is_one_hot=is_one_hot
    )
    descriptor.write_json(processed_data + file_name + ".json")
    descriptor.write_parquet(processed_data + file_name + ".parquet")


def get_different_descriptors(
    features: list = [2, 3, 0], just_mp: bool = False
) -> None:
    """
    Run getting polyhedra descriptors of different types

    Parameters
    ----------
    features: int, optional
        Number or type of expected features:
        2 -> 2 features
        3 -> 3 features
        0 -> structure without size customization (all graphs consist of different number of nodes)
    just_mp: bool, optional
        If yes, then data was obtained only from Materials Project, save with name '..._mp.json'
    """
    for f in features:
        if not just_mp:
            make_poly_descriptor(f, str(f) + "_features")
        else:
            make_poly_descriptor(f, str(f) + "_features_mp")

    is_one_hot = True
    if not just_mp:
        make_poly_descriptor(2, "poly_vector_of_count", is_one_hot)
    else:
        make_poly_descriptor(2, "poly_vector_of_count_mp", is_one_hot)
    print(
        f"Creating {len(features)+1} data with different presents of descriptors for PolyDataset are completed"
    )


def main(just_mp: bool):
    """
    Run total collection

    Parameters
    ----------
    just_mp: bool, optional
        If yes, then data was obtained only from Materials Project, save with name '..._mp.json'
    """
    combine_structure_and_poly(just_mp=just_mp)
    get_different_descriptors(just_mp=just_mp)


if __name__ == "__main__":
    main(True)
