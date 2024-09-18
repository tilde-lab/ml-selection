
import polars as pl
from ml_selection.data_massage.polyhedra import get_poly_elements, get_periodic_number


def element_counter(elements: list) -> dict:
    """
    Counts the number of atoms in a list.
    """
    atmos_dict = {}

    for el in elements:
        if el not in atmos_dict.keys():
            atmos_dict[el] = 1
        else:
            atmos_dict[el] += 1
    return atmos_dict


def compare_to_pauling(poly_for_compare: list, pauling_poly: pl.DataFrame):
    """
    Compares polyhedra for a specific structure (it is assumed that they were obtained
    by independent calculations) with value from Pauling File.
    """
    found = 0
    found_entry = False

    for current_poly in poly_for_compare:
        found_entry = False
        for pauling_line in pauling_poly.rows():
            if pauling_line[5] == current_poly[1]:
                elements = pauling_line[-1]

                elements_pauling = element_counter(get_poly_elements([elements], idx=0))
                current_elements = element_counter([get_periodic_number(el) for el in current_poly[2]])

                if elements_pauling == current_elements:
                    found += 1
                    break
                found_entry = True

            # since they were already previously found in pauling file
            elif found_entry:
                break

    if found_entry == False:
        print('Not found "entry" in PAULING FILE data')
    elif len(poly_for_compare) == found:
        return True
    else:
        print('Not correct poly')


def comparison_of_polyhedra(path_pauling: str, path_domestic: str):
    """
    Collects all the resulting polyhedra for a specific structure
    and checks if they match the values from Pauling File.
    """
    match = 0
    last_entry = None
    temp_poly_store = []

    domestic_poly = pl.read_json(path_domestic)
    pauling_poly = pl.read_json(path_pauling)

    for i, poly in enumerate(domestic_poly.to_numpy().tolist()):
        if poly[1] == last_entry:
            temp_poly_store.append(poly)
        else:
            if temp_poly_store == []:
                last_entry = poly[1]
                temp_poly_store.append(poly)
                continue

            if compare_to_pauling(temp_poly_store, pauling_poly):
                match += 1
                print('Values coincided: ', match, ' From: ', i)

            last_entry = poly[1]
            temp_poly_store = []


if __name__ == "__main__":
    path_pauling = 'ml-selection/data/raw_mpds/large_poly.json'
    path_domestic = 'ml-selection/data/poly/poly_4_seeb_test.json'

    comparison_of_polyhedra(path_pauling, path_domestic)





