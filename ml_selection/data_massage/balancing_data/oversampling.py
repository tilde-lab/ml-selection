import polars as pl
import smogn
from polars import DataFrame


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
        X = pl.read_csv(str_path)
        y = pl.read_csv(seeb_path)
    else:
        X, y = str_dfrm, seebeck_dfrm

    if seeb_path != None:
        atoms = [eval(i) for i in list(X["atom"])]
        distance = [eval(i) for i in list(X["distance"])]
    else:
        atoms = [i for i in list(X["atom"])]
        distance = [i for i in list(X["distance"])]

    total = []

    for i in range(len(atoms)):
        total.append(
            atoms[i] + distance[i] + [list(y.row(i)) for i in range(len(y))][i]
        )

    total_df = pl.DataFrame(total)
    total_df.rename(columns={total_df.columns[-1]: "Seebeck coefficient"}, inplace=True)

    new_data = smogn.smoter(data=total_df, y="Seebeck coefficient", rel_method="auto")

    new_l = []
    seebeck_l = []

    for row in [list(new_data) for i in range(new_data)]:
        atoms = row[:100]
        distance = row[100:200]
        seeb = row[-1]
        new_l.append([atoms, distance])
        seebeck_l.append(seeb)

    df_str = pl.DataFrame(new_l, columns=["atom", "distance"])
    df_seeb = pl.DataFrame(seebeck_l, columns=["Seebeck coefficient"])

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


if __name__ == "__main__":
    total = pl.read_csv(
        "/Users/alina/PycharmProjects/ml-selection/data/processed_data/cut_str.csv",
    )
    seebeck = pl.read_csv(
        "/Users/alina/PycharmProjects/ml-selection/data/processed_data/cut_seeb.csv",
    )
    total = pl.concat([seebeck["Seebeck coefficient"], total], axis=1)
    features = ["atom", "distance"]

    train_size = int(0.9 * len(total))
    test_size = len(total) - train_size

    train_str = total.iloc[:train_size]
    test_str = total.iloc[train_size:]

    train_seebeck = seebeck.iloc[:train_size]
    test_seebeck = seebeck.iloc[train_size:]

    test_str.to_csv(
        "/Users/alina/PycharmProjects/ml-selection/data/processed_data/test_over_str.csv",
        index=False,
    )

    test_seebeck.to_csv(
        "/Users/alina/PycharmProjects/ml-selection/data/processed_data/test_over_seeb.csv",
        index=False,
    )

    make_oversampling(
        str_dfrm=train_str,
        seebeck_dfrm=train_seebeck,
        path_to_save="/Users/alina/PycharmProjects/ml-selection/data/processed_data/",
    )
