import numpy as np
from scipy.spatial import distance
from operator import itemgetter
from ase.data import chemical_symbols
from data.mendeleev_table import get_periodic_number
from metis_backend.metis_backend.datasources.fmt import detect_format
from metis_backend.metis_backend.structures.cif_utils import cif_to_ase


def find_metal(atom=None, coord=None):
    """
    Count the number of metal center atom in complex.

    Parameters
    ----------
    atom : list, None
        Full atomic labels of complex.
    coord : array_like, None
        Full atomic coordinates of complex.

    Returns
    -------
    total_metal : int
        The total number of center atom.
    atom_metal : list
        Atomic labels of center atom.
    coord_metal : ndarray
        Atomic coordinates of  center atom.
    """
    if atom is None or coord is None:
        raise TypeError("find_metal needs two arguments: atom, coord")

    total_metal = 0
    atom_metal = []
    coord_metal = []

    for i in range(len(atom)):
        number = get_periodic_number(atom[i])

        if 21 <= number <= 30 or \
                39 <= number <= 48 or \
                57 <= number <= 80 or \
                89 <= number <= 109:

            total_metal += 1
            atom_metal.append(atom[i])
            coord_metal.append(coord[i])

    coord_metal = np.asarray(coord_metal, dtype=np.float64)

    return total_metal, atom_metal, coord_metal


def extract_poly(atom=None, coord=None, n_center=1, cutoff_ligand=2.8, number_of_atoms=1):
    """
    Search the polyhedron structure in complex and return atoms and coordinates.

    Parameters
    ----------
    atom : list
        Full atomic labels of complex.
    coord : array_like
        Full atomic coordinates of complex.
    n_center : int
        Number of metal atom that will be taken as center atom for
        finding atomic coordinates of polyhedron structure of interest.
        Default value is 1.
    cutoff_ligand : float, optional
        Cutoff distance for screening ligand bond.
        Default value is 2.8.
    number_of_atoms : int, optional
        Number of atoms in polyhedron

    Returns
    -------
    atom_octa : list
        Atomic labels of polyhedron structure.
    coord_polyhedron : ndarray
        Atomic coordinates of polyhedron structure.
    """
    if atom is None or coord is None:
        raise TypeError("needs two arguments: atom and coord")

    # Count the number of metal center atom
    total_metal, atom_metal, coord_metal = find_metal(atom, coord)

    if n_center <= 0:
        raise ValueError("index of metal must be positive integer")

    elif n_center > total_metal:
        raise ValueError("user-defined index of metal is greater than "
                         "the total metal in complex.")

    metal_index = n_center - 1
    dist_list = []

    for i in range(len(list(atom))):
        dist = distance.euclidean(coord_metal[metal_index], coord[i])
        if dist <= cutoff_ligand:
            dist_list.append([atom[i], coord[i], dist])

    # sort list of tuples by distance in ascending order
    dist_list.sort(key=itemgetter(2))

    # Keep only first n atoms
    dist_list = dist_list[:number_of_atoms]
    atom_polyhedron, coord_polyhedron, dist = zip(*dist_list)

    atom_polyhedron = list(atom_polyhedron)
    coord_polyhedron = np.asarray(coord_polyhedron, dtype=np.float64)

    return atom_polyhedron, coord_polyhedron


def get_poly_info(ase_obj):
    atoms_poly, coord_poly = extract_poly(
        [chemical_symbols[i] for i in ase_obj.numbers], ase_obj.positions, number_of_atoms=16
    )
    print(atoms_poly)


def get_poly_type():
    NotImplemented


input_file = '/root/projects/ml-selection/input'
structure = open(input_file).read()
fmt = detect_format(structure)
ase_obj, _ = cif_to_ase(structure)
get_poly_info(ase_obj)
