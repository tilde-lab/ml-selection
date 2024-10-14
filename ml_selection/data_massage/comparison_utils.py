from ml_selection.data_massage.database_handlers.MPDS.mpds_utils import (
    get_structure_with_exist_poly
)
from ml_selection.data_massage.polyhedra.get_poly_from_ase import get_polyhedrons
from ml_selection.data_massage.polyhedra.search_poly import search_poly_by_entry
import chemparse
from os import listdir
from os.path import isfile, join
from random import randrange



def compare_poly(sid: str, api_key: str, num: int):
    """
    Runs a matching check between polhedra from MPDS and own polyhedra.
    
    Parameters
    ----------
    
    api_key : str
        Key from MPDS account
    num : int
        Number of structures for check    
    """
    onlyfiles = [
        f for f in listdir("ml_selection/cif") if isfile(join("ml_selection/cif", f))
    ]
    idxs = [randrange(len(onlyfiles)) for i in range(num)]
    
    comp_poly = 0
    comp_from = 0

    for i in idxs:
        try:
            poly_true, cif = get_structure_with_exist_poly(
                sid, api_key, from_dir=True, entry='S' + onlyfiles[i].replace('.cif', '').split('S')[-1]
            )
            # there are no poly
            if poly_true == []:
                continue
            poly_domestic = get_polyhedrons(structures=cif)
        except:
            continue
        
        if poly_domestic == [] or poly_domestic == None:
            print('Zero domestic poly')
            continue
        
        cnt = 0
        total_poly = len(poly_true[1])
        
        for poly_t in poly_true[1]:
            poly_true_comp = chemparse.parse_formula(poly_t)
            print(poly_true_comp)
            if poly_true_comp in poly_domestic:
                cnt+= 1
                comp_poly += 1
            comp_from += 1
        print('Correct poly in structure:', cnt)
        print('From:', total_poly)

        
    print('========TOTAL RESULT========')
    print('Correct poly in structure:', comp_poly)
    print('From:', comp_from)


