from data.mendeleev_table import get_periodic_number
import re

def get_poly_elements(poly: list) -> list:
    if poly[8] == None:
        return [None]
    formula = re.sub(r"(?<=\w)([A-Z])", r" \1", poly[8])
    composition = formula.replace('<sub>', ' ').replace('</sub>', ' ').split(' ')
    elements = []

    for el in composition:
        if el == '':
            continue
        if not(el.isdigit()):
            try:
               elements.append(get_periodic_number(el))
            except:
                return [None]
        else:
            for i in range(int(el)-1):
                elements.append(elements[-1])

    return elements

def get_int_poly_type(poly: list) -> int:
    values = poly[7].replace('-', ' ').replace('#', ' ').split(' ')
    vertex = int(values[0])*10
    type = ord(values[1]) - ord('a') + 1
    return [vertex , type]

def size_customization(elements, size_v=100) -> list:
    if len(elements) >= size_v:
        return elements[:size_v]

    res = elements.copy()
    while len(res) < size_v:
        res = res + elements

    return res[:size_v]