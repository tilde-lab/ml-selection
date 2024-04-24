"""
Run add polyhedra to structures by entry, make descriptor
"""

from data_massage.data_handler import DataHandler

hand = DataHandler(True)
raw_data = "/root/projects/ml-selection/data/raw_data/"
processed_data = "/root/projects/ml-selection/data/processed_data"
def combine_structure_and_poly():
    dfrm = hand.add_polyhedra(raw_data + "structures.json")
    dfrm.to_json(raw_data + "poly.json", orient="split")

def make_poly_descriptor():
    descriptor = hand.process_polyhedra(raw_data + "poly.json")
    descriptor.to_csv(processed_data + "poly_descriptor.csv")
    
if __name__ == "__main__":
    make_poly_descriptor()
