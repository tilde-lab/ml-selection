"""
Run add polyhedra to structures by entry, make descriptor
"""

from data_massage.data_handler import DataHandler

hand = DataHandler(True)
raw_data = "/root/projects/ml-selection/data/raw_data/"
processed_data = "/root/projects/ml-selection/data/processed_data/poly/"

def combine_structure_and_poly():
    dfrm = hand.add_polyhedra(raw_data + "rep_structures.json")
    dfrm.to_json(raw_data + "large_poly.json", orient="split")

def make_poly_descriptor(features=2, file_name='test', is_one_hot=False):
    descriptor = hand.process_polyhedra(raw_data + "large_poly.json", features=features, is_one_hot=is_one_hot)
    descriptor.to_csv(processed_data + file_name + ".csv", index=False)

def get_different_descriptors():
    features = [2, 3, 0]
    for f in features:
        make_poly_descriptor(f, str(f) + '_features')

    is_one_hot = True
    make_poly_descriptor(2, 'poly_vector_of_count', is_one_hot)
    print(f'Creating {len(features)+1} data with different presents of descriptors for PolyDataset are completed')

if __name__ == "__main__":
    combine_structure_and_poly()
    get_different_descriptors()
