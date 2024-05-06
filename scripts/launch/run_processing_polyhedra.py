"""
Run add polyhedra to structures by entry, make descriptor
"""

from data_massage.data_handler import DataHandler

hand = DataHandler(True)
raw_data = "/root/projects/ml-selection/data/raw_data/"
processed_data = "/root/projects/ml-selection/data/processed_data/poly/"


def combine_structure_and_poly():
    """
    Combine structures and polyhedra by entry
    """
    dfrm = hand.add_polyhedra(raw_data + "rep_structures.json")
    dfrm.to_json(raw_data + "large_poly.json", orient="split")


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
    descriptor.to_csv(processed_data + file_name + ".csv", index=False)


def get_different_descriptors(features: list = [2, 3, 0]) -> None:
    """
    Run getting polyhedra descriptors of different types

    Parameters
    ----------
    features: int, optional
        Number or type of expected features:
        2 -> 2 features
        3 -> 3 features
        0 -> structure without size customization (all graphs consist of different number of nodes)
    """
    for f in features:
        make_poly_descriptor(f, str(f) + "_features")

    is_one_hot = True
    make_poly_descriptor(2, "poly_vector_of_count", is_one_hot)
    print(
        f"Creating {len(features)+1} data with different presents of descriptors for PolyDataset are completed"
    )


def main():
    """
    Run total collection
    """
    combine_structure_and_poly()
    get_different_descriptors()


if __name__ == "__main__":
    main()
