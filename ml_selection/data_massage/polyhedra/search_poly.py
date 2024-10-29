import polars as pl


def search_poly_by_entry(
    entry: str, path_to_poly: str = "ml_selection/structures_props/raw_mpds/aets.csv"
) -> pl.DataFrame:
    """
    Search the polyhedra dataset for a specific entry.

    Args:
        entry (str): The entry to search
        path_to_poly (str, optional): The path to the polyhedra dataset file.
        Defaults to 'ml_selection/structures_props/raw_mpds/aets.csv'.

    Returns:
    polyhedra_data : list
        [entry, [chemical formulas for every exist poly]]
    """
    poly = pl.read_csv(path_to_poly)
    target_poly = poly.filter(pl.col("Entry") == entry)

    polyhedra_data = [entry, []]

    for i in range(len(target_poly)):
        polyhedra_data[1].append(
            target_poly.row(i)[3]
            .replace("<sub>", "")
            .replace("</sub>", "")
            .replace(" ", "")
        )

    return polyhedra_data
