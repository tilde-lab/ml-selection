import time

import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from mpds_client import MPDSDataRetrieval, MPDSDataTypes
from pandas import DataFrame


class RequestMPDS:
    """
    Make requests to MPDS database
    """

    def __init__(self, dtype: int, api_key: str = None) -> None:
        self.client = MPDSDataRetrieval(dtype=dtype, api_key=api_key)
        self.client.chillouttime = 1
        self.dtype = dtype
        self.api_key = api_key
        with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
            self.sid = yaml.load(yamlfile, Loader=yaml.FullLoader)["sid"]
            print("Sid is read successful")

    def make_request(
        self, is_seebeck: bool = False, is_structure: bool = False, phases: list = None
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
                ],
            )

            return answer_df

    @staticmethod
    def make_request_polyhedra(sid: str, entrys: list, phases: list) -> DataFrame:
        """
        Requests information about polyhedra type by https

        Parameters
        ----------
        sid : str
            Sid from browser console (needed authentication)
        entrys : list
            Set with entry value for needed structures
        phases : list
            Phases of structure, should be the same size as entry list

        Returns
        -------
            Answer from MPDS
        """
        # means pages without table Atomic environments
        loss_data = 0
        atomic_data = []

        for i, entry in enumerate(entrys):
            query = f"https://api.mpds.io/v0/download/s?q={entry}&fmt=pdf&sid={sid}"

            response = requests.get(query)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                try:
                    atomic_table = soup.find(
                        "h3", text="Atomic environments"
                    ).find_next("table")
                except:
                    loss_data += 1
                    continue

                rows = atomic_table.find_all("tr")[1:]
                atomic_data.append([phases[i], entry, []])
                for row in rows:
                    cells = row.find_all("td")
                    try:
                        atomic_data[i - loss_data][2].append(
                            [
                                cells[1].text.strip(),
                                cells[2].text.strip(),
                                cells[3].text.strip(),
                            ]
                        )
                    except:
                        continue
                time.sleep(0.1)
            else:
                loss_data += 1
        update_res = [i for i in atomic_data if len(i) == 3]
        res = pd.DataFrame(update_res, columns=["phase_id", "entry", "polyhedra"])
        print(f"Answers without table Atomic environments: {loss_data}")

        return res
