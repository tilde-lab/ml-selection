"""
Main file that starts collecting data for training models. Get median value for Seebeck and structures.
1 phase_id <-> 1 Seebeck <-> many structures.
"""

import polars as pl
import yaml
from polars import DataFrame

from data_massage.calculate_median_value import seebeck_median_value
from data_massage.data_handler import DataHandler
from scripts.launch import run_processing_polyhedra


def get_structures_and_seebeck(
    handler: DataHandler,
    is_uniq_structure_for_phase: bool,
    raw_seebeck_path: str = None,
    raw_str_path: str = None,
    path_to_save: str = None,
) -> DataFrame:
    """
    Get all available Seebeck values from -150 to 200. For each 'phase_id', calculate
    median Seebeck value. Query all available structures for these 'phase_id'.
    Parameters
    ----------
    handler : DataHandler
        class instance
    is_uniq_structure_for_phase : bool
        if True, only 1 Seebeck value is store per 1 structure per 1 'phase_id'
    raw_seebeck_path : str, optional
        Path to file with not processed data for Seebeck value
    raw_str_path : str, optional
        Path to file with not processed data for structures
    path_to_save : str, optional
        Path to folder, where raw data save, appropriate only in the absence
        'raw_seebeck_path' and 'raw_str_path'
    """
    if not raw_seebeck_path:
        # get Seebeck for PEER_REV and AB_INITIO
        seebeck_dfrm = handler.just_seebeck(
            max_value=200, min_value=-150, is_uniq_phase_id=False
        )
        file_path = path_to_save + "seebeck.json"
        seebeck_dfrm.write_json(file_path)
    else:
        seebeck_dfrm = pl.read_json(raw_seebeck_path)

    phases = list(set(seebeck_dfrm["Phase"]))

    # make median Seebeck value
    median_seebeck = seebeck_median_value(seebeck_dfrm, phases)
    file_path = path_to_save + "median_seebeck.json"
    median_seebeck.write_json(file_path)

    if not raw_str_path:
        # get structure and make it ordered
        structures_dfrm = handler.to_order_disordered_str(
            phases=phases, is_uniq_phase_id=is_uniq_structure_for_phase
        )
        file_path = path_to_save + "rep_structures.json"
        structures_dfrm.write_json(file_path)
    else:
        structures_dfrm = pl.read_json(raw_str_path)

    result_dfrm = handler.add_seebeck_by_phase_id(median_seebeck, structures_dfrm)

    if path_to_save:
        csv_file_path = path_to_save + "total.json"
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
    total.write_json(path_to_save + "rep_vect_str_clear.json")

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
    with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
        api_key = yaml.load(yamlfile, Loader=yaml.FullLoader)["api_key"]
        print("Key is read successful")

    raw_path = "/root/projects/ml-selection/data/raw_data/"
    proc_path = "/root/projects/ml-selection/data/processed_data/"
    is_uniq_structure_for_phase = False
    handler = DataHandler(True, api_key)

    result_dfrm = get_structures_and_seebeck(
        handler,
        is_uniq_structure_for_phase,
        raw_seebeck_path=raw_path + "seebeck.json",
        raw_str_path=raw_path + "rep_structures.json",
        path_to_save=raw_path,
    )
    get_data_for_vectors_dataset(handler, result_dfrm, proc_path)
    run_processing_polyhedra.main()


if __name__ == "__main__":
    main()
