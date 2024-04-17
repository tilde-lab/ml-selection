import statistics

import pandas as pd
from pandas import DataFrame


def seebeck_median_value(data: pd.DataFrame, phases: list) -> DataFrame:
    """
    Calculates the median value of Seebeck coefficient from samples with the same 'phase_id'

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame with next columns: 'Phase', 'Formula', 'Seebeck coefficient'
    phases : list
        List with set of phases

    Returns
    -------
        DataFrame with median values of Seebeck coefficient
    """
    new_data_list = []

    for phase in phases:
        seebeck = []
        data_for_phase = [
            string for string in data.values.tolist() if phase == string[0]
        ]

        if len(data_for_phase) == 1:
            new_data_list.append(data_for_phase[0])
            continue

        for value in data_for_phase:
            seebeck.append(value[2])

        median_seebeck = statistics.median(seebeck)

        new_data_for_phase = data_for_phase[0]
        new_data_for_phase[2] = median_seebeck

        new_data_list.append(new_data_for_phase)

    dfrm = pd.DataFrame(
        new_data_list, columns=["phase_id", "Formula", "Seebeck coefficient"]
    )

    return dfrm
