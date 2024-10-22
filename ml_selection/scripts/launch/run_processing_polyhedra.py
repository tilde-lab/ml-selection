"""
Run add polyhedra to structures by entry, make descriptor
"""
import os
import sys
sys.path.append(os.getcwd())


from ml_selection.data_massage.data_handler import DataHandler
import yaml


hand = DataHandler(True)

CONF = "ml_selection/configs/config.yaml"

with open(CONF, "r") as yamlfile:
    conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    raw_data = conf["raw_mpds"]
    processed_data = conf["polyhedra_path"]


def combine_structure_and_poly(just_mp: bool, just_mpds: bool, file_name_with_structures: str, phys_prop: str) -> None:
    """
    Combine structures and polyhedra by entry
    """
    if just_mp:
        dfrm = hand.add_polyhedra(raw_data + f"{file_name_with_structures}.json")
        if phys_prop == "Conductivity":
            dfrm = hand.add_polyhedra(raw_data + "large_poly_mp_conductivity.json")
        else:
            dfrm.write_json(raw_data + "large_poly_mp_seebeck.json")
    elif just_mpds:
        dfrm = hand.add_polyhedra(raw_data + f"{file_name_with_structures}.json")
        if phys_prop == "Conductivity":
            dfrm = hand.add_polyhedra(raw_data + "large_poly_mpds_conductivity.json")
        else:
            dfrm.write_json(raw_data + "large_poly_mpds_seebeck.json")
    else:
        dfrm = hand.add_polyhedra(raw_data + f"{file_name_with_structures}.json")
        if phys_prop == "Conductivity":
            dfrm = hand.add_polyhedra(raw_data + "large_poly_mpds_mp_conductivity.json")
        else:
            dfrm.write_json(raw_data + "large_poly_mpds_mp_seebeck.json")


def make_poly_descriptor(
    file_name: str = "test", phys_prop: str = "Seebeck coefficient"
) -> None:
    """
    Run creating polyhedra descriptor

    Parameters
    ----------
    file_name : str, optional
        Name for result file
    """
    # TODO: add another case: 'large_poly_mpds_mp_seebeck', 'large_poly_mpds_mp_conductivity'; (see func combine_structure_and_poly)
    if phys_prop == "Seebeck coefficient":
        descriptor = hand.process_polyhedra(
            raw_data + "large_poly_mpds_seebeck.json"
        )
    else:
        descriptor = hand.process_polyhedra(
            raw_data + "large_poly_mpds_conductivity.json"
        )
    
    descriptor.write_json(processed_data + file_name + ".json")
    descriptor.write_parquet(processed_data + file_name + ".parquet")


def get_descriptor(phys_prop: str, just_mp: bool = False) -> None:
    """
    Run getting polyhedra descriptor

    Parameters
    ----------
    just_mp: bool, optional
        If yes, then data was obtained only from Materials Project, save with name '..._mp.json'
    """
    if phys_prop == "Conductivity":
        if not just_mp:
            make_poly_descriptor("descriptor_mp_conductivity", phys_prop)
        else:
            make_poly_descriptor("descriptor_mpds_conductivity", phys_prop)
    else:
        if not just_mp:
            make_poly_descriptor("descriptor_mp_seeb", phys_prop)
        else:
            make_poly_descriptor("descriptor_mpds_seeb", phys_prop)

    print(
        f"Creating presents of descriptors for PolyDataset are completed"
    )


def main(just_mp: bool = False, just_mpds: bool = False, mpds_file_name: str = "raw_structures", phys_prop: str = "Conductivity") -> None:
    """
    Run total collection

    Parameters
    ----------
    just_mp: bool, optional
        If yes, then data was obtained only from Materials Project, save with name '..._mp.json'
    just_mpds: bool, optional
        If yes, then data was obtained only from MPDS, load from 'rep_structures_mpds.json'
    mpds_file_name: str, "rep_structures_mpds_seeb"
        Name of file with needed structures
    """
    combine_structure_and_poly(just_mp=just_mp, just_mpds=just_mpds, mpds_file_name=mpds_file_name, phys_prop=phys_prop)
    get_descriptor(just_mp=just_mp, phys_prop=phys_prop)


if __name__ == "__main__":
    main(just_mpds=True, mpds_file_name="rep_structures_mpds_seeb")
