import numpy as np
import polars as pl
import pandas as pd
from mpds_client import MPDSDataRetrieval, MPDSDataTypes


class RequestMPDS:
    """
    Make requests to MplS database
    """

    def __init__(self, dtype: int, api_key: str = None) -> None:
        self.client = MPDSDataRetrieval(dtype=dtype, api_key=api_key)
        self.client.chillouttime = 1
        self.dtype = dtype
        self.api_key = api_key

    def make_request(
        self, is_seebeck: bool = False, is_structure: bool = False, phases: list = None
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
            Answer from MPDS
        """
        if is_seebeck:
            # because .get_dataframe return pd.Dataframe
            dfrm = pd.DataFrame(self.client.get_dataframe({"props": "Seebeck coefficient"}))
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
            # answer_df = pl.from_pandas(pd.DataFrame(
            #     self.client.get_data(
            #         {"props": "atomic structure"},
            #         phases=phases,
            #         fields={
            #             "S": [
            #                 "phase_id",
            #                 "occs_noneq",
            #                 "cell_abc",
            #                 "sg_n",
            #                 "basis_noneq",
            #                 "els_noneq",
            #                 "entry",
            #                 "condition"
            #             ]
            #         },
            #     ),
            #     columns=[
            #         "phase_id",
            #         "occs_noneq",
            #         "cell_abc",
            #         "sg_n",
            #         "basis_noneq",
            #         "els_noneq",
            #         "entry",
            #         "temperature"
            #     ]
            # ))
            answer_df = pl.read_json('/root/projects/ml-selection/data/test_struct.json')
            return answer_df
