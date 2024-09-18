from random import randrange
from request_to_mpds import RequestMPDS


def get_random_s_entry(sid: str, file_with_entry: str = 'structures_props/raw_mpds/s_entries_all.txt') -> tuple:
    """Return random entry with CIF structure and polyhedron
    
    Parameters
    ----------
    sid : str
        Sid from MPDS account
    file_with_entry : str, optional
        File with all avalible entry in MPDS database
        
    Returns
    -------
    (poly : list, cif : str)
        poly - entry, polyhedrons;  cif - crystal structure
    """
    succes = False
    
    while not(succes):
        myfile = open(file_with_entry, mode='r', encoding='utf_8')
        entrys = myfile.read().splitlines() 
        
        idx = randrange(len(entrys))
        entry = entrys[idx]
        
        poly = RequestMPDS.request_polyhedra(sid, [entry.replace(' ', '')])
        
        if poly == []:
            continue
        else:
            cif = RequestMPDS.request_cif(sid, entry.replace(' ', ''))
            if str(cif) != '{"error":"Unknown entry type"}' and cif != None:
                succes = True
    
    return (poly, str(cif))

if __name__ == "__main__":
    poly, cif = get_random_s_entry('SID')
                 
            
        
    
    
    