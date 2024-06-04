"""
Main file that starts collecting data for training models. Get median value for Seebeck and structures.
1 phase_id <-> 1 Seebeck <-> many structures.
"""

import polars as pl
from polars import DataFrame
import yaml

from data_massage.calculate_median_value import seebeck_median_value
from data_massage.data_handler import DataHandler
from scripts.launch import run_processing_polyhedra
from adapter import MPDS_MP_Adapter
from data_massage.database_handlers.MaterialsProject.request_to_mp import RequestMP

CONF = '/root/projects/ml-selection/configs/config.yaml'


def get_structures_and_seebeck(
    handler: DataHandler,
    is_uniq_structure_for_phase: bool,
    raw_seebeck_path: str = None,
    raw_str_path: str = None,
    path_to_save: str = None,
    just_mp: bool = False
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
    just_mp: bool
        if yes, then Seebeck's data will be obtained only from Materials Project
    """
    # get seebeck from MP
    RequestMP(CONF).run_requests()
    phase_id_mp = MPDS_MP_Adapter().run_match_mp_mpds_data()

    if not just_mp:
        if not raw_seebeck_path:
            # get Seebeck for PEER_REV and AB_INITIO from MPDS
            seebeck_dfrm_mpds = handler.just_seebeck(
                max_value=200, min_value=-150, is_uniq_phase_id=False
            )
            file_path = path_to_save + "seebeck.json"
            seebeck_dfrm_mpds.write_json(file_path)
        else:
            seebeck_dfrm_mpds = pl.read_json(raw_seebeck_path)

        # change direction of columns for stack 2 dfrm
        phase_id_mp = phase_id_mp.select(seebeck_dfrm_mpds.columns)
        phases = list(set(seebeck_dfrm_mpds["Phase"])) + list(set(phase_id_mp['Phase']))

        # make median Seebeck value
        median_seebeck = seebeck_median_value(phase_id_mp.vstack(seebeck_dfrm_mpds), phases)
        file_path = path_to_save + "median_seebeck.json"
        median_seebeck.write_json(file_path)
    else:
        phase_id_mp = phase_id_mp.select(['Phase', 'Formula', 'Seebeck coefficient'])
        phases = list(set(phase_id_mp['Phase']))
        median_seebeck = seebeck_median_value(phase_id_mp, phases)
        file_path = path_to_save + "mp_seebeck.json"
        median_seebeck.write_json(file_path)

    if not raw_str_path:
        # get structure and make it ordered
        if not just_mp:
            structures_dfrm = handler.to_order_disordered_str(
                phases=phases, is_uniq_phase_id=is_uniq_structure_for_phase
            )
            file_path = path_to_save + "rep_structures.json"
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

    get_structures_and_seebeck(
        handler,
        is_uniq_structure_for_phase,
        path_to_save=raw_path,
        just_mp=True
    )
    run_processing_polyhedra.main(True)


if __name__ == "__main__":
    main()
