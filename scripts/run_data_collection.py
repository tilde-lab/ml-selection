"""
Main file that starts collecting data for training models. Get median value for Seebeck and structures.
1 phase_id <-> 1 Seebeck <-> many structures.
"""
import pandas as pd
import yaml
from pandas import DataFrame

from data_massage.balancing_data.oversampling import make_oversampling
from data_massage.balancing_data.undersampling import make_undersampling
from data_massage.calculate_median_value import seebeck_median_value
from data_massage.data_handler import DataHandler


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
        seebeck_dfrm.to_json(file_path, orient="split")
    else:
        seebeck_dfrm = pd.read_json(raw_seebeck_path, orient="split")

    phases = set(seebeck_dfrm["Phase"].tolist())
    # make median Seebeck value
    median_seebeck = seebeck_median_value(seebeck_dfrm, phases)

    if not raw_str_path:
        # get structure and make it ordered
        structures_dfrm = handler.to_order_disordered_str(
            phases=phases, is_uniq_phase_id=is_uniq_structure_for_phase
        )
        file_path = path_to_save + "rep_structures.json"
        structures_dfrm.to_json(file_path, orient="split")
    else:
        structures_dfrm = pd.read_json(raw_str_path, orient="split")

    result_dfrm = handler.add_seebeck_by_phase_id(median_seebeck, structures_dfrm)

    if path_to_save:
        csv_file_path = path_to_save + "total.csv"
        result_dfrm.to_csv(csv_file_path, index=False)

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
        dfrm_str.to_csv(path_to_save + "rep_vect_str.csv", index=False)
        dfrm_seeb.to_csv(path_to_save + "rep_vect_seebeck.csv", index=False)

    return dfrm_str, dfrm_seeb


if __name__ == "__main__":
    with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
        api_key = yaml.load(yamlfile, Loader=yaml.FullLoader)['api_key']
        print("Key is read successful")

    raw_path = "/root/projects/ml-selection/data/raw_data/"
    processed_path = "/root/projects/ml-selection/data/processed_data/"

    is_uniq_structure_for_phase = False
    handler = DataHandler(True, api_key)

    str_seeb_dfrm = get_structures_and_seebeck(
        handler,
        is_uniq_structure_for_phase,
        raw_seebeck_path=raw_path+'seebeck.json',
        # raw_str_path=raw_path+'structure.json',
        path_to_save=raw_path
    )
    dfrm_str, dfrm_seeb = convert_structure_to_vectors(
        handler,
        str_seeb_dfrm,
        path_to_save=processed_path,
    )

    df_structs, df_seebeck = make_undersampling(
        str_dfrm=dfrm_str, seebeck_dfrm=dfrm_seeb
    )
    df_structs.to_csv(
        "/root/projects/ml-selection/data/processed_data/under_str.csv",
        index=False,
    )

    df_seebeck.to_csv(
        "/root/projects/ml-selection/data/processed_data/under_seeb.csv",
        index=False,
    )




