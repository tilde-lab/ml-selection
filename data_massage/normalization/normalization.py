import pickle
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def make_normalization(
    data: DataFrame,
    path: str,
    scaler_name: str = "scaler",
    column_name: str = "Seebeck coefficient",
) -> DataFrame:
    """
    Makes data normalization in range (-1, 1).
    Data for normalization is specified as the name of a specific column from the dataframe.
    Return dataframe.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    d = scaler.fit_transform(data[[column_name]].values.reshape(-1, 1))

    data[column_name] = d

    with open(path + scaler_name + ".pkl", "wb") as f:
        pickle.dump(scaler, f)

    return data
