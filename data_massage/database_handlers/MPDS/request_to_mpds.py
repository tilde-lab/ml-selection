import json

import pandas as pd
import polars as pl
from mpds_client import MPDSDataRetrieval, MPDSDataTypes
import yaml
from yaml import Loader

CONF = '/root/projects/ml-selection/configs/config.yaml'


class RequestMPDS:
    """
    Make requests to MPDS database
    """

    def __init__(self, dtype: int = 7, api_key: str = None) -> None:
        conf = open(CONF, 'r')
        self.conf = yaml.load(conf, Loader)
        if api_key == None:
            api_key = self.conf['api_key']
        self.raw_mpds = self.conf['raw_mpds']
        self.client = MPDSDataRetrieval(dtype=dtype, api_key=api_key)
        self.client.chillouttime = 1
        self.dtype = dtype
        self.api_key = api_key

    def make_request(
        self, is_seebeck: bool = False, is_structure: bool = False, is_phase: bool = False,
            phases: list = None, formulas: list = None, sg: list = None, mp_ids: list = None
    ) -> pl.DataFrame:
        """
        Requests data from the MplS according to the input parameters

        Parameters
        ----------
        is_seebeck : bool
            True, if you need to request a Seebeck coefficient
        is_structure : bool
            True, if you need to request a structure
        phases : list, optional
            Phases of structure, if you need request structures for specific phases

        Returns
        -------
            Answer from MPDS:
            If is_structure -> columns: "phase_id", "formula", "occs_noneq", "cell_abc", "sg_n", "basis_noneq",
            "els_noneq", "entry", "temperature"
            If is_seebeck -> "Phase", "Formula", "Seebeck coefficient"
        """
        if is_seebeck:
            # because .get_dataframe return pd.Dataframe
            dfrm = pd.DataFrame(
                self.client.get_dataframe({"props": "Seebeck coefficient"})
            )
            dfrm = pl.from_pandas(dfrm)
            dfrm = dfrm.filter(pl.col("Phase").is_finite())
            dfrm = dfrm.rename({"Value": "Seebeck coefficient"})
            columns_to_drop = [dfrm.columns[i] for i in [2, 3, 4, 5]]
            dfrm = dfrm.drop(columns_to_drop)
            return dfrm

        elif is_structure:
            self.client = MPDSDataRetrieval(
                dtype=MPDSDataTypes.PEER_REVIEWED, api_key=self.api_key
            )
            try:
                answer_df = pl.read_json(self.raw_mpds + 'raw_structures.json')
            except:
                answer_df = pl.from_pandas(
                    pd.DataFrame(
                        self.client.get_data(
                            {"props": "atomic structure"},
                            phases=phases,
                            fields={
                                "S": [
                                    "phase_id",
                                    "chemical_formula",
                                    "occs_noneq",
                                    "cell_abc",
                                    "sg_n",
                                    "basis_noneq",
                                    "els_noneq",
                                    "entry",
                                    "condition",
                                ]
                            },
                        ),
                        columns=[
                            "phase_id",
                            "formula",
                            "occs_noneq",
                            "cell_abc",
                            "sg_n",
                            "basis_noneq",
                            "els_noneq",
                            "entry",
                            "temperature",
                        ],
                    )
                )
            answer_df.write_json(self.raw_mpds + 'raw_structures.json')
            return answer_df

        elif is_phase:
            phase_ids = []
            found, loss = 0, 0
            try:
                with open(self.raw_mpds + 'mpds_phases_jan2024.json', 'r') as file:
                    data = json.load(file)

                # Dict for fast search, with formula and space group
                full_formula_dict = {}
                short_formula_dict = {}
                for row in data:
                    key_full = (row['formula']['full'], row['spg'])
                    key_short = (row['formula']['short'], row['spg'])

                    if key_full not in full_formula_dict:
                        full_formula_dict[key_full] = row['id'].split('/')[-1]

                    if key_short not in short_formula_dict:
                        short_formula_dict[key_short] = row['id'].split('/')[-1]

                # Search match
                for i in range(len(formulas)):
                    key_full = (formulas[i], int(sg[i]))
                    key_short = (formulas[i], int(sg[i]))

                    if key_full in full_formula_dict.keys():
                        phase_ids.append([full_formula_dict[key_full], mp_ids[i], key_full[0], key_full[1]])
                    elif key_short in short_formula_dict:
                        phase_ids.append([short_formula_dict[key_short], mp_ids[i], key_short[0], key_short[1]])

                print('Found matches:', len(phase_ids))
                return pl.DataFrame(phase_ids, schema=['phase_id', 'identifier', 'formula', 'symmetry'])

            except:
                print('Raw data with MPDS phase_ids not found in directory. Start requests!')
                for i in range(len(formulas)):
                    try:
                        self.client.chillouttime = 2
                        ans = self.client.get_data(
                            {"sgs": sg[i], "formulae": formulas[i]}
                        )
                        phase_ids.append([str(ans[0][0]), mp_ids[i]])
                        found += 1
                    except Exception as e:
                        print(e)
                        if e != 'HTTP error code 204: No Results (No hits)':
                            self.client.chillouttime += 1
                        loss += 1
                        print('Not found:', loss)

            print('Matches by formula and Space group found:', found)
            return pl.DataFrame(phase_ids, schema=['phase_id', 'identifier'])




