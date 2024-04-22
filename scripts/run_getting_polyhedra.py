"""
Run collection values from Atomic environments table by https request
"""

import pandas as pd
import yaml

from data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS

with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
    conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    sid, path_to_save = conf["sid"], conf["raw_polyhedra_path"]
    print("Sid is read successful")
str_dfrm = pd.read_json(
    "/root/projects/ml-selection/data/raw_data/structures.json", orient="split"
)
phases, entrys = (
    str_dfrm["phase_id"].values.tolist(),
    str_dfrm["entry"].values.tolist(),
)
res = RequestMPDS.make_request_polyhedra(sid, entrys, phases)
res.to_json(path_to_save, orient="split", index=False)
