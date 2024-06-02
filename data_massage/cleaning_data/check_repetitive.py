import polars as pl
import yaml
from yaml import Loader


CONF = '/root/projects/ml-selection/configs/config.yaml'


def check_same_features(dfrm: pl.DataFrame, columns: list) -> list:
    flag = False
    if type(dfrm) == pl.DataFrame:
        dfrm = dfrm.to_pandas()
        flag = True
    duplics = len(dfrm) - len(dfrm.drop_duplicates(subset=columns))

    if flag == True:
        dfrm = dfrm.drop_duplicates(subset=columns)
        return pl.from_pandas(dfrm)
    return [duplics, dfrm.drop_duplicates(subset=columns)]


def check_disagreements(path_mpds: str = None, path_mp: str = None, from_conf: bool = True):
    """
    Checks differences in Seebeck values between data from MPDS and MP
    """
    conf = open(CONF, 'r')
    conf = yaml.load(conf, Loader)
    if from_conf:
        path_mp, path_mpds = conf['mp_path'], conf['raw_mpds']

    phases = pl.read_json(path_mp + 'id_matches_mp_mpds.json')
    mp_seebeck = pl.read_json(path_mp + 'seebeck_sg_mp.json').rename({'Seebeck coefficient': 'Seebeck mp'})
    mpds_seebeck = pl.read_json(path_mpds + 'seebeck.json').rename({'Phase': 'phase_id', 'Seebeck coefficient': 'Seebeck mpds'})
    phases = phases.with_columns(pl.col("phase_id").cast(pl.Int64))
    all = mpds_seebeck.join(phases, on="phase_id", how="inner")
    all = all.join(mp_seebeck, on="identifier", how="inner").drop('Formula')
    all.write_json(path_mp + 'comparison_mp_mpds.json')

    same_values = []

    # check how many
    all_in_dict = all.to_dicts()
    for row in all_in_dict:
        if int(row['Seebeck mp']['n']['value']) == int(row['Seebeck mpds']):
            same_values.append(row)
        if int(row['Seebeck mp']['p']['value']) == int(row['Seebeck mpds']):
            if row not in same_values:
                same_values.append(row)

    print(f'Found {len(same_values)} same values of Seebeck')
    print(f'{len(all_in_dict) - len(same_values)} examples are different')
    print(pl.DataFrame(same_values))


if __name__ == "__main__":
    check_disagreements()


