from ml_selection.data_massage.database_handlers.MPDS.mpds_utils import (
    get_structure_with_exist_poly
)
from ml_selection.data_massage.polyhedra.get_poly_from_ase import get_polyhedrons
import chemparse


def compare_poly(api_key: str, num: int):
    """
    Runs a matching check between polhedra from MPDS and own polyhedra.
    
    Parameters
    ----------
    
    api_key : str
        Key from MPDS account
    num : int
        Number of structures for check    
    """
    comp_poly = 0
    comp_from = 0

    for i in range(num):
        try:
            poly_true, cif = get_structure_with_exist_poly(api_key)
            poly_domestic = get_polyhedrons(structures=cif)
        except:
            continue
        
        cnt = 0
        total_poly = len(poly_true[0][1])
        
        for poly_t in poly_true[0][1]:
            poly_true_comp = chemparse.parse_formula(poly_t[2])
            if poly_true_comp in poly_domestic:
                cnt+= 1
                comp_poly += 1
            comp_from += 1
        print('Correct poly in structure:', cnt)
        print('From:', total_poly)

        
    print('========TOTAL RESULT========')
    print('Correct poly in structure:', comp_poly)
    print('From:', comp_from)

compare_poly("KEY", 20)

