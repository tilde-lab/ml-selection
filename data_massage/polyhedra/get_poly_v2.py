from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.local_env import JmolNN
import numpy as np

def find_coordination_polyhedra(cif_file):
    structure = Structure.from_file(cif_file)

    voronoi_nn = VoronoiNN()
    
    polyhedra = {}
    
    for site_index, site in enumerate(structure):
        neighbors_info = voronoi_nn.get_nn_info(structure, site_index)
        neighbors = [neighbor['site'].specie.symbol for neighbor in neighbors_info]
        distances = [neighbor['weight'] for neighbor in neighbors_info]
        
        polyhedra[site_index] = {
            'element': site.specie.symbol,
            'coords': site.coords,
            'neighbors': neighbors,
            'distances': distances
        }
    
    return polyhedra

def main():
    cif_file = 'cif/monoclinic_tricliniс/Na[C2O2N3H3POS][H2O]_MPDS_S310420.cif'
    polyhedra = find_coordination_polyhedra(cif_file)
    
    for idx, info in polyhedra.items():
        print(f"Site {idx}:")
        print(f"  Element: {info['element']}")
        print(f"  Coordinates: {info['coords']}")
        print(f"  Neighbors: {info['neighbors']}")
        print(f"  Distances: {info['distances']}")
        print()

if __name__ == "__main__":
    main()
