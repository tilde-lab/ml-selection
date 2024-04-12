import pandas as pd
from pandas import DataFrame


def make_undersampling(
    str_path: str = None,
    seeb_path: str = None,
    str_dfrm: DataFrame = None,
    seebeck_dfrm: DataFrame = None,
):
    """
    Reduces the number of structures per Seebeck value to 50, pre-rounds Seebeck value to a whole number.
    """
    if str_path and seeb_path:
        str_dfrm = pd.read_csv(str_path)
        seebeck_dfrm = pd.read_csv(seeb_path)

    data = pd.concat(
        [seebeck_dfrm["Seebeck coefficient"], str_dfrm], axis=1
    ).values.tolist()

    new_list_str = []
    new_list_seeb = []
    seeb_used = []

    cnt = 0

    # leave only no more than 50 structures for each Seebeck
    for row in sorted(data):
        if float(round(row[0])) not in seeb_used:
            cnt = 1
            seeb_used.append(float(round(row[0])))
            new_list_seeb.append(float(round(row[0])))
            new_list_str.append([row[1], row[2]])
        else:
            if cnt >= 50:
                cnt += 1
                continue
            else:
                new_list_seeb.append(float(round(row[0])))
                new_list_str.append([row[1], row[2]])
                cnt += 1

    seeb_df = pd.DataFrame(new_list_seeb, columns=["Seebeck coefficient"])
    str_df = pd.DataFrame(new_list_str, columns=["atom", "distance"])
    shuffle_df = (
        pd.concat([seeb_df, str_df], axis=1).sample(frac=1).reset_index(drop=True)
    )

    df_seebeck = shuffle_df.iloc[:, :1]
    df_structs = shuffle_df.iloc[:, 1:]

    return df_structs, df_seebeck
