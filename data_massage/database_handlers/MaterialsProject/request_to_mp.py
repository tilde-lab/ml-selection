# See here https://contribs-api.materialsproject.org/#/contributions/queryContributions
import yaml
from yaml import Loader
from mpcontribs.client import Client
import json
from data_massage.data_handler import DataHandler


class RequestMP:
    """
    Make requests to Materials Project database
    """

    def __init__(self, conf_path) -> None:
        conf = open(conf_path, 'r')
        dictionary = yaml.load(conf, Loader)
        self.api_key = dictionary['api_mp']
        self.mp_path = dictionary['mp_path']

    def request_seebeck(self):
        """
        Make requests to "carrier_transport" project, get "identifier", "data.S.n.value", "data.S.p.value".
        Save answer in json format
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
            return cont_in_dict


if __name__ == "__main__":
    path = '/root/projects/ml-selection/configs/config.yaml'
    client = RequestMP(path)
    seebeck_dict = client.request_seebeck()
    handler = DataHandler()
    dataframe = handler.convert_mp_data_to_dataframe(seebeck_dict)
    print(dataframe)
    dataframe.write_json(client.mp_path + "seebeck_mp_dfrm.json")

