import pickle

import polars as pl
from polars import DataFrame
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yaml


def make_normalization(
    data: DataFrame,
    scaler_name: str = "scaler",
) -> DataFrame:
    """
    Makes data normalization in range (-1, 1).
    Data for normalization is specified as the name of a specific column from the dataframe.
    Return dataframe.
    """
    with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
        yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
        path = yaml_f["scaler_path"]

    exceptions_col = ['phase_id', 'Phase', 'Formula']
    for column_name in data.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if column_name not in exceptions_col:
            try:
                scaler.fit([list(i) for i in list(data[column_name])])
                d = scaler.transform([list(i) for i in list(data[column_name])])
                data = data.with_columns(pl.Series(column_name, d.tolist()))
            except:
                scaler.fit(np.array([i for i in list(data[column_name])]).reshape(-1, 1))
                d = scaler.transform(np.array([i for i in list(data[column_name])]).reshape(-1, 1))
                d = [i[0] for i in d]
                data = data.with_columns(pl.Series(column_name, d))
            with open(path + scaler_name + column_name + ".pkl", "wb") as f:
                pickle.dump(scaler, f)

    return data
