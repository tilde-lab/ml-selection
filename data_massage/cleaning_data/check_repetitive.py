import pandas as pd
import polars as pl


def check_same_features(dfrm, columns: list) -> list:
    flag = False
    if type(dfrm) == pl.DataFrame:
        dfrm = dfrm.to_pandas()
        flag = True
    duplics = len(dfrm) - len(dfrm.drop_duplicates(subset=columns))

    if flag == True:
        dfrm = dfrm.drop_duplicates(subset=columns)
        return pl.from_pandas(dfrm)
    return [duplics, dfrm.drop_duplicates(subset=columns)]


if __name__ == "__main__":
    struct = pd.read_json('/root/projects/ml-selection/data/raw_data/rep_structures.json', orient='split')
    seeb = pd.read_json('/root/projects/ml-selection/data/raw_data/median_seebeck.json', orient='split')

    total = pd.merge(struct, seeb, on="phase_id", how="inner")
    num, dfrm = check_same_features(total, columns=['entry'])

    print(f'Number of removed duplicates: {num}')
    pl.from_pandas(dfrm).write_json('/root/projects/ml-selection/data/processed_data/rep_vect_str_clear.json')
