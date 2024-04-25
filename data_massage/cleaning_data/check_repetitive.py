import pandas as pd

def check_same_features(dfrm: pd.DataFrame, columns: list) -> list:
    duplics = len(dfrm) - len(dfrm.drop_duplicates(subset=columns))
    return [duplics, dfrm.drop_duplicates(subset=columns)]


if __name__ == "__main__":
    struct = pd.read_csv('/root/projects/ml-selection/data/processed_data/rep_vect_str.csv')
    seeb = pd.read_csv('/root/projects/ml-selection/data/processed_data/rep_vect_seebeck.csv')

    total = pd.concat([seeb, struct], axis=1)
    num, dfrm = check_same_features(total, columns=['atom', 'distance'])
    print(f'Number of removed duplicates: {num}')

    dfrm.to_csv('/root/projects/ml-selection/data/processed_data/rep_vect_str_clear.csv')
