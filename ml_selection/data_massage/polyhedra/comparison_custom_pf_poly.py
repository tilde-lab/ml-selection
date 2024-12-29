from ml_selection.data_massage.database_handlers.MPDS.mpds_utils import get_structure_with_exist_poly
from ml_selection.data_massage.polyhedra.get_poly_from_cif import run_getting_polyhedrons
from ml_selection.data_massage.polyhedra.poly_description import polyhedra_types
from pymatgen.core.composition import Composition

import os

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

def delete_all_files_in_directory(directory_path):
    try:
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):  
                os.remove(file_path)    
        print(f"All files in {directory_path} have been deleted.")
    except Exception as e:
        print(f"Error while deleting files: {e}")

def clean_repeat(poly: list) -> list:
    """
    Remove repeat polyhedra
    """
    temp = []
    for i in poly:
        if i not in temp:
            temp.append(i)
    return temp


def formula_to_dict_pymatgen(formula):
    comp = Composition(formula)
    return {str(el): int(amount) for el, amount in comp.get_el_amt_dict().items()}


def run_comparison(api_key: str, sid: str, n_iteration: int = 10, type_extraction: str = "custom") -> None:
    """
    Compare random polyhedra from MPDS with custom polyhedra
    """
    store_acc = []
    
    for i in range(n_iteration):
        delete_all_files_in_directory("ml_selection/cif")
        poly, _ = get_structure_with_exist_poly(sid, api_key)
        try:
            custom_poly = run_getting_polyhedrons(type_extraction=type_extraction, path_to_dir="ml_selection/cif/")
        except:
            print('CIF in invalid format, polyhedron definition ended in error')
            continue
        poly_true = [[i[1], formula_to_dict_pymatgen(i[2])] for i in clean_repeat(poly[0][1])]
        
        try:
            poly_pred = [[polyhedra_types[key[0]], key[1]] for key in clean_repeat(list(custom_poly.items())[-1][1])]
        except:
            print('The found polyhedron does not exist in the annotation PF. Probable cause: too many vertices')
            store_acc.append(0)
            continue
        
        cnt, succ = 0, 0
        for pred in poly_pred:
            cnt += 1
            if pred in poly_true:
                succ += 1
        
        if succ > 0:
            store_acc.append(100/cnt*succ)
            print('Percentage of correct polyhedra: ', round(100/cnt*succ, 2))
        else:
            store_acc.append(0)
            print('Percentage of correct polyhedra: ', 0)
        
    print('Average percentage of correct polyhedra: ', sum(store_acc)/len(store_acc))
        
run_comparison('KEY', 'SID')