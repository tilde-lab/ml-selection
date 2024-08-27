from metis_backend.metis_backend.datasources.fmt import detect_format
from metis_backend.metis_backend.structures.cif_utils import cif_to_ase
from metis_backend.metis_backend.structures.struct_utils import poscar_to_ase, optimade_to_ase

from ase import Atoms
from ase.neighborlist import NeighborList

import numpy as np
import polars as pl


def extract_poly(atom=None, coord=None, cell=None, cutoff=3.0):
    """
    Extract all polyhedrons in the structure by finding neighboring atoms within the cutoff distance.

    Parameters
    ----------
    atom : list
        Full atomic labels
    coord : array
        Full atomic coordinates
    cell : array
        Lattice vectors of the unit cell
    cutoff : float, optional
        Cutoff distance for finding neighboring atoms.
        Default value is 3.0

    Returns
    -------
    polyhedrons : list of dict
        List of dictionaries containing atomic labels and coordinates of polyhedrons
    """
    if atom is None or coord is None or cell is None:
        raise TypeError("extract_poly needs three arguments: atom, coord, and cell")

    atoms = Atoms(symbols=atom, positions=coord, cell=cell, pbc=True)

    # atomic cut-off radius
    cutoffs = [cutoff] * len(atoms)

    # make list with all neighbors in current cut-off
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    polyhedrons = []

    for i, center_atom in enumerate(atom):
        # offsets - shifts of neighboring atoms if they are outside cell
        indices, offsets = nl.get_neighbors(i)

        # for each neighbor found, its coordinates and distance to the central atom
        neighbor_coords = atoms.positions[indices] + np.dot(offsets, atoms.get_cell())
        distances = np.linalg.norm(neighbor_coords - atoms.positions[i], axis=1)

        if len(indices) > 0:  # if found neighbors
            polyhedron = {
                "center_atom": center_atom,
                "center_coord": atoms.positions[i],
                "atoms": [center_atom] + [atom[j] for j in indices],
                "coords": [atoms.positions[i].tolist()] + neighbor_coords.tolist(),
                "distances": [0.0] + distances.tolist()  # 0.0 for central atom
            }
            polyhedrons.append(polyhedron)

    return polyhedrons


def get_polyhedrons(path_structures: str, path_to_save: str, cutoff=3.0):
    poly_store = []

    structures = open(path_structures).read()
    fmt = detect_format(structures)

    if fmt == "cif":
        atoms, error = cif_to_ase(structures)
        if error:
            raise RuntimeError(error)
    elif fmt == "poscar":
        atoms, error = poscar_to_ase(structures)
        if error:
            raise RuntimeError(error)

    elif fmt == "optimade":
        atoms, error = optimade_to_ase(structures, skip_disorder=True)
        if error:
            raise RuntimeError(error)

    # if the data is received from mpds
    else:
        structures = pl.read_json(path_structures)
        for structure in structures.rows():
            atoms = Atoms(
                symbols=structure[4],
                positions=structure[3],
                cell=structure[1]
            )

            # found all available polyhedrons for that structure
            polyhedrons = extract_poly(
                atom=[symbol for symbol in atoms.get_chemical_symbols()],
                coord=atoms.get_positions(),
                cell=atoms.get_cell(),
                cutoff=cutoff
            )

            for poly in polyhedrons:
                poly_store.append(
                    [structure[0], structure[5], poly["atoms"], poly["coords"], poly["distances"]]
                )

    if fmt in ["cif", "poscar", "optimade"]:
        # found all available polyhedrons for that structure
        polyhedrons = extract_poly(
            atom=[symbol for symbol in atoms.get_chemical_symbols()],
            coord=atoms.get_positions(),
            cell=atoms.get_cell(),
            cutoff=cutoff
        )

        for poly in polyhedrons:
            poly_store.append(
                [atoms[0], atoms[5], poly["atoms"], poly["coords"], poly["distances"]]
            )

    res = pl.DataFrame(poly_store, schema=["phase_id", "entry", "atoms", "coords", "distances"]).write_json(path_to_save)
    print(res)


if __name__ == "__main__":
    path = 'ml-selection/data/raw_mpds/rep_structures_mpds_seeb.json'
    path_to_save = 'ml-selection/data/poly/poly_4_seeb.json'

    get_polyhedrons(path, path_to_save)
