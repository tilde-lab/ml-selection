import statistics

import polars as pl
from polars import DataFrame


def seebeck_median_value(data: pl.DataFrame, phases: list) -> DataFrame:
    """
    Calculates the median value of Seebeck coefficient from samples with the same 'phase_id'

    Parameters
    ----------
    data : polars DataFrame
        DataFrame with next columns: 'Phase', 'Formula', 'Seebeck coefficient'
    phases : list
        Set of phases

    Returns
    -------
        DataFrame with median values of Seebeck coefficient
    """
    new_data_list = []

    for phase in phases:
        seebeck = []
        data_for_phase = data.filter(pl.col("Phase") == phase)
        data_for_phase_list = [
            list(data_for_phase.row(i)) for i in range(len(data_for_phase))
        ]

        if len(data_for_phase_list) == 1:
            new_data_list.append(data_for_phase_list[0])
            continue

        for value in data_for_phase_list:
            seebeck.append(value[2])

        median_seebeck = statistics.median(seebeck)

        new_data_for_phase = data_for_phase_list[0]
        new_data_for_phase[2] = median_seebeck

        new_data_list.append(new_data_for_phase)

    dfrm = pl.DataFrame(
        new_data_list, schema=["phase_id", "Formula", "Seebeck coefficient"]
    )

    return dfrm
