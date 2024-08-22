import re

from data.mendeleev_table import get_periodic_number


def get_poly_elements(poly: list, idx: int = 9) -> list:
    """
    Get periodic element number for each atom in a polyhedra

    Parameters
    ----------
    poly : list
        Data with atoms name
    idx: int
        Number of index with elements info

    Returns
    -------
    elements  : list
         Periodic numbers of elements
    """
    if poly[idx] == None:
        return [None]
    formula = re.sub(r"(?<=\w)([A-Z])", r" \1", poly[idx])
    composition = formula.replace("<sub>", " ").replace("</sub>", " ").split(" ")
    elements = []

    for el in composition:
        if el == "":
            continue
        if not (el.isdigit()):
            try:
                elements.append(get_periodic_number(el))
            except:
                return [None]
        else:
            for i in range(int(el) - 1):
                elements.append(elements[-1])

    return elements


def size_customization(elements: list, size_v: int = 100) -> list:
    """
    Add required number of elements to get the desired size of list

    Parameters
    ----------
    elements  : list
        Include of periodic numbers of elements in polyhedra
    size_v : int
        Number of elements in final list

    Returns
    -------
    res : list
        Data in desired size
    """
    if len(elements) >= size_v:
        return elements[:size_v]

    res = elements.copy()
    while len(res) < size_v:
        res = res + elements

    return res[:size_v]
