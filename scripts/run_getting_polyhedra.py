"""
Run add polyhedra to structures by entry
"""

from data_massage.data_handler import DataHandler

hand = DataHandler(True)
dfrm = hand.add_polyhedra("/root/projects/ml-selection/data/raw_data/structures.json")
dfrm.to_json("/root/projects/ml-selection/data/raw_data/poly.json", orient="split")
