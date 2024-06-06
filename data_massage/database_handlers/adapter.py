import yaml
from yaml import Loader
import polars as pl
from data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS
import pandas as pd

CONF = "/root/projects/ml-selection/configs/config.yaml"


class MPDS_MP_Adapter:
    """
    Adapting data from Materials Project to MPDS.
    Find 'phase_id' from MPDS by formula, space group from MP
    """

    def __init__(self):
        conf = open(CONF, "r")
        self.conf = yaml.load(conf, Loader)
        self.mp_path, self.raw_mpds = self.conf["mp_path"], self.conf["raw_mpds"]
        self.db = ["MP", "MPDS"]
        self.mp_dfrm = pl.read_json(self.mp_path + "seebeck_sg_mp.json")
        self.mpds_client = RequestMPDS()

    def finding_matches_id_by_formula_sg(self, is_all_id: bool = False) -> pl.DataFrame:
        """
        Find 'phase_id' for material from MPDS by formula, space group from Materials Project.
        Save answer in JSON-format

        Returns
        -------
        phases: pl.DataFrame
            Consist of columns: 'phase_id' (id from MPDS),
            'identifier' (id from Materials Project)
        is_all_id: bool
            Will all IDs from MP be processed? If 'True': all
        """
        try:
            if is_all_id:
                phases = pl.read_json(self.mp_path + 'id_matches_mp_mpds_all.json')
            else:
                phases = pl.read_json(self.mp_path + 'id_matches_mp_mpds.json')
        except:
            if is_all_id:
                self.mp_dfrm = pl.read_json(self.mp_path + 'all_id_mp.json')
            phases = self.mpds_client.make_request(
                is_phase=True, sg=list(self.mp_dfrm['symmetry']), formulas=list(self.mp_dfrm['formula']),
                mp_ids=list(self.mp_dfrm['identifier'])
            )
            if is_all_id:
                phases.write_json(self.mp_path + 'id_matches_mp_mpds_all.json')
            else:
                phases.write_json(self.mp_path + 'id_matches_mp_mpds.json')
        return phases

    def match_structures_by_phase_id(self, phases: list) -> pl.DataFrame:
        """
        Make request to MPDS by phase_id, get structures

        Parameters
        ----------
        phases : list
            'phase_id' from MPDS that correspond to 'identifier' from MP

        Returns
        -------
        structures_for_mp_seebeck: pl.DataFrame
            Structures from MPDS
        """
        try:
            structures_for_mp_seebeck = pl.read_json(
                self.mp_path + "structures_mp_mpds.json"
            )
        except:
            structures_for_mp_seebeck = self.mpds_client.make_request(
                is_structure=True, phases=phases
            )
            structures_for_mp_seebeck.write_json(
                self.mp_path + "structures_mp_mpds.json"
            )
        return structures_for_mp_seebeck


    def process_seebeck_to_mpds_format(self, seebeck_dfrm_mpds_format):
        seebeck_list = seebeck_dfrm_mpds_format["Seebeck coefficient"]
        new_column_seeb = []
        for row in seebeck_list:
            new_column_seeb.append(row["n"]["value"])

        seebeck_dfrm_mpds_format = seebeck_dfrm_mpds_format.with_columns(
            pl.Series("Seebeck coefficient", new_column_seeb)
        )

        # filter by value
        condition = (seebeck_dfrm_mpds_format["Seebeck coefficient"] >= -150) & (
            seebeck_dfrm_mpds_format["Seebeck coefficient"] <= 200
        )
        # use filter
        filtered_df = seebeck_dfrm_mpds_format.filter(condition)
        filtered_df = filtered_df.with_columns(pl.col("phase_id").cast(pl.Int64))

        return filtered_df

    def run_match_mp_mpds_data(self):
        phases = self.finding_matches_id_by_formula_sg()
        seebeck_dfrm_mpds_format = pl.from_pandas(
            pd.merge(
                self.mp_dfrm.to_pandas(),
                phases.to_pandas(),
                on="identifier",
                how="inner",
            )
        ).drop(columns=["identifier", "symmetry"])
        seebeck_dfrm_mpds_format = self.process_seebeck_to_mpds_format(
            seebeck_dfrm_mpds_format
        )
        return seebeck_dfrm_mpds_format.rename(
            {"phase_id": "Phase", "formula": "Formula"}
        )


if __name__ == "__main__":
    structures_for_mp_seebeck = MPDS_MP_Adapter().run_match_mp_mpds_data()
