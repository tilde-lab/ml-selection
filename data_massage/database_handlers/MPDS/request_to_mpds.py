import numpy as np
import pandas as pd
from mpds_client import MPDSDataRetrieval, MPDSDataTypes
from pandas import DataFrame


class RequestMPDS:
    """
    Makes requests to MPDS database.
    """

    def __init__(self, dtype: int, api_key: str = None) -> None:
        self.client = MPDSDataRetrieval(dtype=dtype, api_key=api_key)
        self.client.chillouttime = 1
        self.dtype = dtype
        self.api_key = api_key

    def make_request(
        self, is_seebeck: bool = False, is_structure: bool = False, phases: list = None
    ) -> DataFrame:
        """
        Requests data from the MPDS according to the input parms.
        Return DataFrame or dict.
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
                ],
            )

            return answer_df
