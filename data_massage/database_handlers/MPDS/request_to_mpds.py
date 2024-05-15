import numpy as np
import polars as pd
import yaml
from mpds_client import MPDSDataRetrieval, MPDSDataTypes
from polars import DataFrame


class RequestMPDS:
    """
    Make requests to MPDS database
    """

    def __init__(self, dtype: int, api_key: str = None) -> None:
        self.client = MPDSDataRetrieval(dtype=dtype, api_key=api_key)
        self.client.chillouttime = 1
        self.dtype = dtype
        self.api_key = api_key

    def make_request(
        self, is_seebeck: bool = False, is_structure: bool = False, phases: list = None, is_temperature: bool = False
    ) -> DataFrame:
        """
        Requests data from the MPDS according to the input parameters

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
            dfrm = self.client.get_dataframe({"props": "Seebeck coefficient"})
            dfrm = dfrm[np.isfinite(dfrm["Phase"])]
            dfrm.rename(columns={"Value": "Seebeck coefficient"}, inplace=True)
            dfrm.drop(dfrm.columns[[2, 3, 4, 5]], axis=1, inplace=True)
            return dfrm

        elif is_structure:
            self.client = MPDSDataRetrieval(
                dtype=MPDSDataTypes.PEER_REVIEWED, api_key=self.api_key
            )
            answer_df = pd.DataFrame(
                self.client.get_data(
                    {"props": "atomic structure"},
                    phases=phases,
                    fields={
                        "S": [
                            "phase_id",
                            "occs_noneq",
                            "cell_abc",
                            "sg_n",
                            "basis_noneq",
                            "els_noneq",
                            "entry",
                            "condition"
                        ]
                    },
                ),
                columns=[
                    "phase_id",
                    "occs_noneq",
                    "cell_abc",
                    "sg_n",
                    "basis_noneq",
                    "els_noneq",
                    "entry",
                    "temperature"
                ],
            )
            return answer_df
