import pandas as pd
import smogn
from pandas import DataFrame


def make_oversampling(
    str_path: str = None,
    seeb_path: str = None,
    str_dfrm: DataFrame = None,
    seebeck_dfrm: DataFrame = None,
    path_to_save: str = None,
) -> tuple:
    """
    Make oversampling by SMOTER algorithm

    Parameters
    ----------
    str_path : str, optional
        Path to structures
    seeb_path : str, optional
        Path to Seebeck coefficient
    str_dfrm : DataFrame, optional
        DataFrame with next columns: 'atom', 'distance'
    seebeck_dfrm : DataFrame, optional
        DataFrame with Seebeck coefficient
    path_to_save : str, optional
        Path to folder for save result

    Returns
    -------
    (df_str, df_seeb)
       A tuple of 2 DataFrames, first consist of atoms and distance, second - Seebeck value
    """
    if str_path and seeb_path:
        X = pd.read_csv(str_path)
        y = pd.read_csv(seeb_path)
    else:
        X, y = str_dfrm, seebeck_dfrm

    atoms = [eval(i) for i in X["atom"].values.tolist()]
    distance = [eval(i) for i in X["distance"].values.tolist()]
    total = []

    for i in range(len(atoms)):
        total.append(atoms[i] + distance[i] + y.values.tolist()[i])

    total_df = pd.DataFrame(total)
    total_df.rename(columns={total_df.columns[-1]: "Seebeck coefficient"}, inplace=True)

    new_data = smogn.smoter(data=total_df, y="Seebeck coefficient", rel_method="auto")

    new_l = []
    seebeck_l = []

    for row in new_data.values.tolist():
        atoms = row[:100]
        distance = row[100:200]
        seeb = row[-1]
        new_l.append([atoms, distance])
        seebeck_l.append(seeb)

    df_str = pd.DataFrame(new_l, columns=["atom", "distance"])
    df_seeb = pd.DataFrame(seebeck_l, columns=["Seebeck coefficient"])

    if path_to_save:
        df_str.to_csv(
            f"{path_to_save}over_str.csv",
            index=False,
        )
        df_seeb.to_csv(
            f"{path_to_save}over_seeb.csv",
            index=False,
        )
    return (df_str, df_seeb)
