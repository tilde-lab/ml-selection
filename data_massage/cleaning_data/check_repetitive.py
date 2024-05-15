import polars as pl

def check_same_features(dfrm: pl.DataFrame, columns: list) -> list:
    duplics = len(dfrm) - len(dfrm.drop_duplicates(subset=columns))
    return [duplics, dfrm.drop_duplicates(subset=columns)]

if __name__ == "__main__":
    struct = pl.read_json('/root/projects/ml-selection/data/raw_data/rep_structures.json', orient='split')
    seeb = pl.read_json('/root/projects/ml-selection/data/raw_data/median_seebeck.json', orient='split')

    total = pl.merge(struct, seeb, on="phase_id", how="inner")
    num, dfrm = check_same_features(total, columns=['entry'])
    print(f'Number of removed duplicates: {num}')

    dfrm.to_csv('/root/projects/ml-selection/data/processed_data/rep_vect_str_clear.csv', index=False)
