"""
Run add polyhedra to structures by entry, make descriptor
"""

from data_massage.data_handler import DataHandler

hand = DataHandler(True)
raw_data = "/root/projects/ml-selection/data/raw_data/"
processed_data = "/root/projects/ml-selection/data/processed_data/"

def combine_structure_and_poly():
    dfrm = hand.add_polyhedra(raw_data + "rep_structures.json")
    dfrm.to_json(raw_data + "large_poly.json", orient="split")

def make_poly_descriptor():
    descriptor = hand.process_polyhedra(raw_data + "large_poly.json", features=3)
    descriptor.to_csv(processed_data + "3_features_poly_descriptor.csv", index=False)
    
if __name__ == "__main__":
    combine_structure_and_poly()
    make_poly_descriptor()
