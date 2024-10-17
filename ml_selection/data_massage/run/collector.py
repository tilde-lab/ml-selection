"""
Main file that starts collecting data for training models. Get median value for properties and structures.
1 phase_id <-> 1 Seebeck <-> many structures.
"""

import polars as pl
from polars import DataFrame
import yaml

from ml_selection.data_massage.calculate_median_value import phys_prop_median_value
from ml_selection.data_massage.data_handler import DataHandler
from ml_selection.scripts.launch import run_processing_polyhedra
from ml_selection.data_massage.database_handlers.adapter import MPDS_MP_Adapter
from ml_selection.data_massage.database_handlers.MaterialsProject.request_to_mp import RequestMP

CONF = "ml_selection/configs/config.yaml"


def get_structures_and_phys_prop(
    handler: DataHandler,
    is_uniq_structure_for_phase: bool,
    phys_prop: str = "Seebeck coefficient",
    raw_prop_path: str = None,
    raw_str_path: str = None,
    path_to_save: str = None,
    just_mp: bool = False,
    just_mpds: bool = True,
    min_value: int = None,
    max_value: int = None,
    not_clean_not_ordered_str: bool = True,
) -> DataFrame:
    """
    Get all available properties. For each 'phase_id', calculate
    median value. Query all available structures for these 'phase_id'.
    Available getting values for Seebeck coefficient, thermal conductivity. MPDS supports both properties,
    MP just conductivity.


    Parameters
    ----------
    handler : DataHandler
        class instance
    is_uniq_structure_for_phase : bool
        if True, only 1 Seebeck value is store per 1 structure per 1 'phase_id'
    phys_prop : str
        Property name (available: Seebeck coefficient, thermal conductivity)
    raw_prop_path : str, optional
        Path to file with not processed data for property
    raw_str_path : str, optional
        Path to file with not processed data for structures
    path_to_save : str, optional
        Path to folder, where raw data save, appropriate only in the absence
        'raw_seebeck_path' and 'raw_str_path'
    just_mp: bool
        if yes, then Seebeck's data (available for Seebeck coefficient)
        will be obtained only from Materials Project
    just_mpds: bool
        if yes, then properties (Seebeck coefficient, thermal conductivity)
        will be obtained only from MPDS
    not_clean_not_ordered_str: bool
        Return not clean and not ordered structures. Can be useful if the structures
        themselves are not needed, but an entry is needed to connect property with polyhedra.
    """
    # get seebeck from MP
    if not just_mpds:
        RequestMP(CONF).run_requests()
        phase_id_mp = MPDS_MP_Adapter().run_match_mp_mpds_data()

    if not just_mp:
        if not raw_prop_path:
            # get property for PEER_REV from MPDS
            phys_prop_dfrm_mpds = handler.just_phys_prop(
                max_value=max_value,
                min_value=min_value,
                is_uniq_phase_id=False,
                phys_prop=phys_prop,
            )
            if phys_prop == "Seebeck coefficient":
                file_path = path_to_save + "seebeck.json"
            else:
                file_path = path_to_save + "conductivity.json"
            phys_prop_dfrm_mpds.write_json(file_path)
        else:
            phys_prop_dfrm_mpds = pl.read_json(raw_prop_path)

        if not just_mpds:
            # change direction of columns for stack 2 dfrm
            phase_id_mp = phase_id_mp.select(phys_prop_dfrm_mpds.columns)
            phases = list(set(phys_prop_dfrm_mpds["Phase"])) + list(
                set(phase_id_mp["Phase"])
            )

            # make median value for property
            median_phys_prop = phys_prop_median_value(
                phase_id_mp.vstack(phys_prop_dfrm_mpds), phases
            )
        else:
            phases = list(set(phys_prop_dfrm_mpds["Phase"]))
            median_phys_prop = phys_prop_median_value(phys_prop_dfrm_mpds, phases)
        if phys_prop == "Seebeck coefficient":
            file_path = path_to_save + "median_seebeck_mpds.json"
        else:
            file_path = path_to_save + "median_conductivity.json"
        median_phys_prop.write_json(file_path)

    elif not just_mpds:
        # NOT SUPPORT for conductivity, just for seebeck
        phase_id_mp = phase_id_mp.select(["Phase", "Formula", "Seebeck coefficient"])
        phases = list(set(phase_id_mp["Phase"]))
        median_phys_prop = phase_id_mp
        file_path = path_to_save + "mp_seebeck.json"
        median_phys_prop.write_json(file_path)

    if not raw_str_path:
        # get structure and make it ordered (if needed)
        if not just_mp:
            structures_dfrm = handler.to_order_disordered_str(
                phases=phases,
                is_uniq_phase_id=is_uniq_structure_for_phase,
                return_not_clean_not_ordered=not_clean_not_ordered_str,
            )
            if phys_prop == "Seebeck coefficient":
                file_path = path_to_save + "rep_structures_mpds_seeb_not_clean.json"
            else:
                file_path = path_to_save + "rep_structures_mpds_conductivity.json"
            structures_dfrm.write_json(file_path)
        else:
            try:
                file_path = path_to_save + "mp_structures.json"
                structures_dfrm = pl.read_json(file_path)
            except:
                structures_dfrm = handler.to_order_disordered_str(
                    phases=phases, is_uniq_phase_id=is_uniq_structure_for_phase
                )
                file_path = path_to_save + "mp_structures.json"
                structures_dfrm.write_json(file_path)
    else:
        structures_dfrm = pl.read_json(raw_str_path)

    result_dfrm = handler.add_phys_prop_by_phase_id(median_phys_prop, structures_dfrm)

    if path_to_save:
        csv_file_path = path_to_save + "total_mpds.json"
        result_dfrm.write_json(csv_file_path)

    return result_dfrm


def convert_structure_to_vectors(
    handler: DataHandler, dfrm: DataFrame, path_to_save: str = None
) -> DataFrame:
    """
    Converts to a structure in a format of 2 vectors. In the first vector, periodic number of atom is stored,
    in the second, the distance from the origin of coordinates.
    Saves only the first 100 atoms falling within a given distance.
    """
    dfrm_str, dfrm_seeb = handler.to_cut_vectors_struct(dfrm=dfrm)

    if path_to_save:
        dfrm_str.write_json(path_to_save + "rep_vect_str.json")
        dfrm_seeb.write_json(path_to_save + "rep_vect_seebeck.json")

    total = pl.concat((dfrm_str, dfrm_seeb), how="horizontal")
    total.write_json(path_to_save + "total_str_seeb.json")

    return dfrm_str, dfrm_seeb


def get_data_for_vectors_dataset(
    handler: DataHandler, str_seeb_dfrm: DataFrame, processed_path: str
):
    """
    Get data in format for VectorsGraphDataset
    """
    convert_structure_to_vectors(
        handler,
        str_seeb_dfrm,
        path_to_save=processed_path,
    )


def main():
    """
    Launch data collection step by step
    """
    with open(CONF, "r") as yamlfile:
        conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
        api_key = conf["api_key"]
        raw_path = conf["raw_mpds"]

    is_uniq_structure_for_phase = False
    handler = DataHandler(True, api_key)

    get_structures_and_phys_prop(
        handler,
        is_uniq_structure_for_phase,
        path_to_save=raw_path,
        just_mpds=True,
        phys_prop="Seebeck coefficient",
        # # ! uncomment if you want to use not ordered structure (for getting poly)
        # raw_str_path='/root/projects/ml-selection/data/raw_mpds/rep_structures_mpds_seeb_not_clean.json',
        # will use ordered structures for getting poly (by 'entry')
        raw_str_path="ml_selection/structures_props/raw_mpds/raw_structures.json",
        raw_prop_path="ml_selection/structures_props/raw_mpds/seebeck.json",
        min_value=-150,
        max_value=200,
    )
    run_processing_polyhedra.main(just_mpds=True)


if __name__ == "__main__":
    main()
