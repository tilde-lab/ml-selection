# See here https://contribs-api.materialsproject.org/#/contributions/queryContributions
import yaml
from yaml import Loader
from mpcontribs.client import Client
from mp_api.client import MPRester
from data_massage.data_handler import DataHandler
import json
import polars as pl


class RequestMP:
    """
    Make requests to Materials Project database
    """

    def __init__(self, conf_path) -> None:
        conf = open(conf_path, 'r')
        dictionary = yaml.load(conf, Loader)
        self.api_key = dictionary['api_mp']
        self.mp_path = dictionary['mp_path']

    def request_seebeck(self) -> list:
        """
        Make requests to "carrier_transport" project, get values of Seebeck coefficient
        Save answer in json format

        Returns
        -------
        data: list
            Consist of dicts, where 1 dict <-> 1 sample from db.
            Keys: "identifier", "formula", "data"
        """
        query = {
            "project": "carrier_transport",
            "_limit": 500,  # maximum 500 per request
            "_fields": [
                "identifier", "formula", "data.S.n.value", "data.S.p.value"
            ],
        }
        contributions = []
        has_more, page = True, 1

        try:
            with open(self.mp_path + 'seebeck_mp.json') as f:
                data = json.load(f)
                print("Data is present in the directory, no request needed")
                return data
        except:
            with Client(self.api_key, project="carrier_transport") as client:
                while has_more:
                    resp = client.contributions.queryContributions(
                        page=page, **query
                    ).result()
                    contributions += resp["data"]
                    has_more = resp["has_more"]
                    page += 1
                    print(f'Get result from page: {page}')
            print(f'Received samples: {len(contributions)}')
            f = open(self.mp_path + 'seebeck_mp.json', 'w')
            cont_in_dict = json.dumps(contributions)
            f.write(cont_in_dict)
            print(f'Successfully saved!')
            return eval(cont_in_dict)

    def request_space_group(self, ids, step=120) -> pl.DataFrame:
        """
        Request symmetry by material_id. Retrieves number of spacy group

        Parameters
        ----------
        ids : list
            Material ids

        Returns
        -------
        dfrm: pl.DataFrame
            Consist of columns: 'identifier', 'symmetry'
        """
        try:
            dfrm = pl.read_json(self.mp_path + 'space_group_mp.json')
            print("Space groups are present in the directory, no request needed")
            return dfrm
        except:
            client = MPRester(self.api_key)
            ids_in_batch = [ids[i:i + step] for i in range(0, len(ids), step)]
            ans_ids, ans_sg = [], []

            for batch in ids_in_batch:
                answer = client.summary.search(material_ids=batch, fields=['material_id', 'symmetry'])
                [(ans_ids.append(str(i.material_id)), ans_sg.append(i.symmetry.number)) for i in answer]
            dfrm = pl.DataFrame({'identifier': ans_ids, 'symmetry': ans_sg}, schema=['identifier', 'symmetry'])
            dfrm.write_json(self.mp_path + 'space_group_mp.json')
            return dfrm

    def run_requests(self, step: int = 120) -> pl.DataFrame:
        """
        Make requests of Seebeck coefficient and Space group. Save answer in json format

        Parameters
        ----------
        step : int
            Number of material ids in query. Maximum 120

        Returns
        -------
        dfrm: pl.DataFrame
            Consist of columns: 'identifier', 'formula', 'Seebeck coefficient' 'symmetry'
        """
        seebeck_dict = self.request_seebeck()
        handler = DataHandler(False)
        dfrm_seebeck = handler.convert_mp_data_to_dataframe(seebeck_dict)
        sp_gr = self.request_space_group(list(dfrm_seebeck['identifier']), step=step)
        dfrm = handler.add_seebeck_by_phase_id(dfrm_seebeck, sp_gr)
        dfrm.write_json(self.mp_path + 'seebeck_sg_mp.json')
        return dfrm


if __name__ == "__main__":
    path = '/root/projects/ml-selection/configs/config.yaml'
    RequestMP(path).run_requests()



