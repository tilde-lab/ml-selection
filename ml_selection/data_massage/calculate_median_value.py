import statistics

import polars as pl
from polars import DataFrame


def phys_prop_median_value(data: pl.DataFrame, phases: list) -> DataFrame:
    """
    Calculates the median value of property from samples with the same 'phase_id'

    Parameters
    ----------
    data : polars DataFrame
        DataFrame with next columns: 'Phase', 'Formula', 'NAME_OF_PROPERTY'
    phases : list
        Set of phases

    Returns
    -------
        DataFrame with median values of physical property values
    """
    new_data_list = []

    for phase in phases:
        property = []
        data_for_phase = data.filter(pl.col("Phase") == phase)
        data_for_phase_list = [
            list(data_for_phase.row(i)) for i in range(len(data_for_phase))
        ]

        if len(data_for_phase_list) == 1:
            new_data_list.append(data_for_phase_list[0])
            continue

        for value in data_for_phase_list:
            property.append(value[2])

        median_property = statistics.median(property)

        new_data_for_phase = data_for_phase_list[0]
        new_data_for_phase[2] = median_property

        new_data_list.append(new_data_for_phase)

    dfrm = pl.DataFrame(new_data_list, schema=data.columns)

    return dfrm
