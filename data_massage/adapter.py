import yaml
from yaml import Loader
import polars as pl
from data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS

CONF = '/root/projects/ml-selection/configs/config.yaml'


class MPDS_MP_Adapter():
    """
    Adapting data from Materials Project to MPDS.
    Find 'phase_id' from MPDS by formula, space group from MP
    """
    def __init__(self):
        conf = open(CONF, 'r')
        self.conf = yaml.load(conf, Loader)
        self.mp_path, self.raw_mpds = self.conf['mp_path'], self.conf['raw_mpds']
        self.db = ['MP', 'MPDS']
        self.mp_dfrm = pl.read_json(self.mp_path + 'seebeck_sg_mp.json')
        self.mpds_client = RequestMPDS()

    def finding_matches_id_by_formula_sg(self) -> pl.DataFrame:
        """
        Find 'phase_id' for material from MPDS by formula, space group from Materials Project.
        Save answer in JSON-format

        Returns
        -------
        phases: pl.DataFrame
            Consist of columns: 'phase_id' (id from MPDS),
            'identifier' (id from Materials Project)
        """
        try:
            phases = pl.read_json(self.mp_path + 'id_matches_mp_mpds.json')
        except:
            phases = self.mpds_client.make_request(
                is_phase=True, sg=list(self.mp_dfrm['symmetry']), formulas=list(self.mp_dfrm['formula']),
                mp_ids=list(self.mp_dfrm['identifier'])
            )
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
            structures_for_mp_seebeck = pl.read_json(self.raw_mpds + 'structures_mp_mpds.json')
        except:
            structures_for_mp_seebeck = self.mpds_client.make_request(
                is_structure=True, phases=phases
            )
            structures_for_mp_seebeck.write_json(self.mp_path + 'structures_mp_mpds.json')
        return structures_for_mp_seebeck

    def run_match_mp_mpds_data(self):
        phases = self.finding_matches_id_by_formula_sg()
        structures_for_mp_seebeck = self.match_structures_by_phase_id(list(phases['phase_id']))
        print('Number of found structures in MPDS for Seebeck from MP:', len(structures_for_mp_seebeck))


if __name__ == "__main__":
    phases = MPDS_MP_Adapter().run_match_mp_mpds_data()




