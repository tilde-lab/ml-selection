from data_massage.database_handlers.adapter import MPDS_MP_Adapter
from data_massage.database_handlers.MaterialsProject.request_to_mp import RequestMP


def get_and_match_ids():
    """
    Run request all IDs from Materials Project, then match it with IDs from MPDS
    """
    RequestMP().request_all_data()
    MPDS_MP_Adapter().finding_matches_id_by_formula_sg(is_all_id=True)


if __name__ == "__main__":
    get_and_match_ids()
