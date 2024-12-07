polyhedra_properties = {
            "1#a": {"vertices": 1, "aet": [{'1': {}}], "atom_inside": True},
            "2#a": {"vertices": 2, "aet": [{'2': {}}], "atom_inside": True},
            "2#b": {"vertices": 2, "aet": [{'2': {}}], "atom_inside": True},
            "3#a": {"vertices": 3, "aet": [{'3': {}}], "atom_inside": True},
            "3#b": {"vertices": 3, "aet": [{'3': {}}], "atom_inside": False},
            "4-a": {"vertices": 4, "aet": [{'4': {}}], "atom_inside": True}, 
            "4#b": {"vertices": 4, "aet": [{'4': {}}], "atom_inside": False},  # tetrahedron; central atom outside
            "4#c": {"vertices": 4, "aet": [{'4': {}}], "atom_inside": True},  # central atom outside
            "4#d": {"vertices": 4, "aet": [{'4': {}}], "atom_inside": False},  # central atom outside
            "5-a": {"vertices": 5, "aet": [{'4': {'3': 2, '4': 1}}, {'1': {'3': 4}}]},
            "5-c": {"vertices": 5, "aet": [{'3': {'3': 4}}, {'2': {'3': 3}}]},
            "5#d": {"vertices": 5, "aet": [{'5': {}}], "atom_inside": False},  # central atom outside
            "5#e": {"vertices": 5, "aet": [{'5': {}}], "atom_inside": True, "coplanar": True},
            "6-a": {"vertices": 6, "aet": [{'6': {'3': 4}}]},
            "6-b": {"vertices": 6, "aet": [{'6': {'3': 1, '4': 2}}]},
            "6-d": {"vertices": 6, "aet": [{'5': {'3': 2, '5': 1}}, {1: {'3': 5}}]},
            "6-h": {"vertices": 6, "aet": [{'2': {'3': 4}}, {'2': {'3': 2, '4': 1}}, {'2': {'3': 3, '4': 1}}]},
            "6#c": {"vertices": 6, "aet": [{'6': {}}], "coplanar": True},  # coplanar hexagon
            "6#e": {"vertices": 6, "aet": [{'6': {}}], "atom_inside": False, "coplanar": True},  # non-coplanar hexagon
            "7-a": {"vertices": 7, "aet": [{'3': {'3': 5}}, {'3': {'3': 4}}, {'1': {'3': 3}}]},
            "7-b": {"vertices": 7, "aet": [{'3': {'3': 5}}, {'3': {'3': 3}}, {'1': {'3': 6}}]},
            "7-c": {"vertices": 7, "aet": [{'3': {'3': 2, '4': 2}}, {'3': {'3': 1, '4': 2}}, {'1': {'3': 3}}]},
            "7-d": {"vertices": 7, "aet": [{'4': {'3': 3, '4': 1}}, {'2': {'3': 3}}, {'1': {'3': 6}}]},
            "7-e": {"vertices": 7, "aet": [{'6': {'3': 2, '6': 1}}, {'1': {'3': 6}}]},
            "7-f": {"vertices": 7, "aet": [{'3': {'3': 5}}, {'3': {'3': 3}}, {'1': {'3': 6}}]},
            "7-g": {"vertices": 7, "aet": [{'3': {'3': 5}}, {'3': {'3': 4}}, {'1': {'3': 3}}]},
            "7-h": {"vertices": 7, "aet": [{'5': {'3': 4}}, {'2': {'3': 5}}]},
            
            "8-a": {"vertices": 8, "aet": [{'8': {'4': 3}}]},
            "8-b": {"vertices": 8, "aet": [{'8': {'3': 3, '4': 1}}]},
            "8-c": {"vertices": 8, "aet": [{'6': {'3': 4}}, {'2': {'3': 6}}]},
            "8-d": {"vertices": 8, "aet": [{'4': {'3': 5}}, {'4': {'3': 4}}]},
            
            "8-e": {"vertices": 8, "aet": [{'2': {'3': 4, '4': 1}}, {'2': {'3': 3, '4': 1}}, {'2': {'3': 1, '4': 2}}, {'1': {'3': 5}}, {'1': {'3': 3}}]},
            "8-f": {"vertices": 8, "aet": [{'6': {'3': 2, '4': 2}}, {'2': {'3': 3}}]},
            "9-a": {"vertices": 9, "aet": [{'6': {'3': 5}}, {'3': {'3': 4}}]},
            "9-b": {"vertices": 9, "aet": [{'4': {'3': 2, '4': 2}}, {'3': {'4': 3}}, {'1': {'3': 4}}]},
            "9-c": {"vertices": 9, "aet": [{'3': {'3': 6}}, {'3': {'3': 5}},  {'3': {'3': 3}}]},
            "9-d": {"vertices": 9, "aet": [{'6': {'3': 1, '4': 1, '6': 1}}, {'3': {'3': 2, '4': 2}}]},
            "9-e": {"vertices": 9, "aet": [{'4': {'3': 5}}, {'4': {'3': 4}},  {'1': {'3': 6}}]},
            
            "10-a": {"vertices": 10, "aet": [{'6': {'3': 4}}, {'4': {'3': 3}}]},
            "10-b": {"vertices": 10, "aet": [{'6': {'3': 5}}, {'3': {'3': 4}}, {'1': {'3': 6}}]},
            "10-c": {"vertices": 10, "aet": [{'8': {'3': 5}}, {'2': {'3': 4}}]},
            "10-d": {"vertices": 10, "aet": [{'4': {'3': 5}}, {'4': {'3': 4}}, {'2': {'3': 6}}]},
            "10-e": {"vertices": 10, "aet": [{'4': {'3': 4, '4': 1}}, {'3': {'3': 3, '4': 1}}, {'2': {'3': 1, '4': 2}}]},
            "10-h": {"vertices": 10, "aet": [{'6': {'3': 3, '4': 1}}, {'3': {'3': 2, '4': 2}}, {'1': {'3': 6}}]},
            "10-i": {"vertices": 10, "aet": [{'2': {'3': 5, '4': 1}}, {'2': {'3': 3, '4': 1}}, {'2': {'3': 2, '4': 2}}, {'2': {'3': 1, '4': 2}}, {'1': {'3': 5}}, {'1': {'3': 3}}]},
            "10-l": {"vertices": 10, "aet": [{'3': {'3': 6}}, {'3': {'3': 5}}, {'3': {'3': 4}}, {'1': {'3': 3}}]},
            "11-a": {"vertices": 11, "aet": [{'9': {'3': 2, '4': 2}}, {'2': {'4': 3}}]},
            "11-b": {"vertices": 11, "aet": [{'8': {'3': 5}}, {'2': {'3': 4}}, {'1': {'3': 5}}]},

            "11-d": {"vertices": 11, "aet": [{'4': {'3': 4, '4': 1}}, {'2': {'3': 3, '4': 1}}, {'2': {'3': 2, '4': 2}}, {'2': {'3': 1, '4': 2}},  {'1': {'3': 6}}]},
            "11-e": {"vertices": 11, "aet": [{'4': {'3': 6}}, {'3': {'3': 5}}, {'3': {'3': 4}}, {'1': {'3': 3}}]},
            "11-f": {"vertices": 11, "aet": [{'6': {'3': 5}}, {'3': {'3': 3}}, {'1': {'3': 9}}]},
            "11-g": {"vertices": 11, "aet": [{'4': {'3': 2, '4': 2}}, {'4': {'3': 1, '4': 3}}, {'2': {'3': 1, '4': 2}}, {'1': {'3': 4}}]},
            "11-i": {"vertices": 11, "aet": [{'5': {'3': 4}}, {'3': {'3': 5}}, {'2': {'3': 6}}, {'1': {'3': 7}}]},
            "12-a": {"vertices": 12, "aet": [{'12': {'3': 5}}]}, 
            "12-b": {"vertices": 12, "aet": [{'12': {'3': 2, '4': 2}}]}, #same!
            "12-d": {"vertices": 12, "aet": [{'12': {'3': 2, '4': 2}}]}, #same!
            "12-c": {"vertices": 12, "aet": [{'10': {'3': 2, '4': 2}}, {'2': {'3': 5}}]}, 
            "12-f": {"vertices": 12, "aet": [{'12': {'4': 2, '6': 1}}]}, 
            "12-g": {"vertices": 12, "aet": [{'8': {'3': 2, '4': 2}}, {'2': {'3': 1, '4': 3}}, {'2': {'4': 3}}]}, 
            "12-i": {"vertices": 12, "aet": [{'8': {'3': 5}}, {'2': {'3': 6}}, {'2': {'3': 4}}]}, 
            "12-j": {"vertices": 12, "aet": [{'4': {'3': 6}}, {'3': {'3': 5}}, {'2': {'3': 4}}, {'2': {'3': 3}}, {'1': {'3': 7}}]}, 
            "12-k": {"vertices": 12, "aet": [{'4': {'3': 3, '4': 2}}, {'2': {'3': 4, '4': 1}}, {'2': {'3': 2, '4': 1}}, {'2': {'3': 1, '4': 2}}, {'1': {'3': 6}}, {'1': {'3': 4}}]}, 
            "13-a": {"vertices": 13, "aet": [{'10': {'3': 5}}, {'2': {'3': 6}}, {'1': {'3': 4}}]}, 
            "13-b": {"vertices": 13, "aet": [{'11': {'3': 2, '4': 2}}, {'2': {'3': 4, '4': 1}}]}, 
            "13-c": {"vertices": 13, "aet": [{'4': {'3': 4, '4': 1}}, {'4': {'3': 2, '4': 2}}, {'2': {'3': 5}}, {'2': {'3': 4}}, {'1': {'3': 6}}]},
            "13-d": {"vertices": 13, "aet": [{'6': {'3': 4, '4': 1}}, {'3': {'3': 3}}, {'3': {'3': 2, '4': 2}}, {'1': {'3': 9}}]}, 
            "13-e": {"vertices": 13, "aet": [{'6': {'3': 1, '4': 2}}, {'3': {'3': 4, '4': 2}}, {'3': {'3': 3, '4': 2}}, {'1': {'3': 3}}]}, 
            "13-g": {"vertices": 13, "aet": [{'6': {'3': 4, '4': 1}}, {'2': {'3': 3, '4': 1}}, {'2': {'3': 2, '4': 2}}]},
            "13-j": {"vertices": 13, "aet": [{'8': {'3': 2, '4': 1, '5': 1}}, {'3': {'3': 4}}, {'2': {'3': 4, '5': 1}}]},
            "14-a": {"vertices": 14, "aet": [{'12': {'3': 5}}, {'2': {'3': 6}}]},  
            "14-b": {"vertices": 14, "aet": [{'8': {'4': 3}}, {'6': {'4': 4}}]},  
            "14-c": {"vertices": 14, "aet": [{'6': {'3': 5}}, {'4': {'3': 4, '4': 1}}, {'4': {'3': 2, '4': 2}}]},  
            "14-d": {"vertices": 14, "aet": [{'12': {'3': 2, '4': 2}}, {'2': {'3': 6}}]},  
            "14-e": {"vertices": 14, "aet": [{'6': {'3': 5}}, {'6': {'3': 3}}, {'2': {'3': 9}}]},  
            "14-f": {"vertices": 14, "aet": [{'6': {'3': 3, '4': 1}}, {'3': {'3': 5}}, {'2': {'3': 7}}, {'2': {'3': 5, '4': 1}}, {'1': {'3': 3}}]},
            "14-g": {"vertices": 14, "aet": [{'6': {'3': 3, '5': 1}}, {'4': {'3': 2, '4': 1, '5': 1}}, {'2': {'3': 6}}, {'2': {'3': 5}}]},  
            "15-a": {"vertices": 15, "aet": [{'12': {'3': 5}}, {'3': {'3': 6}}]},
            # "15-d": {"vertices": 15, "edges": 47, "polygon": [72, 108]},  # equatorial 5-capped pentagonal prism
            "15-c": {"vertices": 15, "aet": [{'8': {'3': 6}}, {'5': {'3': 4}}, {'2': {'3': 5}}]},  
            "15-d": {"vertices": 15, "aet": [{'6': {'3': 3, '5': 1}}, {'4': {'3': 4, '5': 1}}, {'2': {'3': 6}}, {'2': {'3': 5}}, {'1': {'3': 4}}]}, 
            "15-e": {"vertices": 15, "aet": [{'4': {'3': 7}}, {'2': {'3': 5}}, {'2': {'3': 4, '4': 1}}, {'2': {'3': 3, '4': 1}}, {'2': {'3': 3}}, {'2': {'3': 2, '4': 2}}, {'1': {'3': 4}}]}, 
            "16-a": {"vertices": 16, "aet": [{'12': {'3': 5}}, {'4': {'3': 6}}]},
            # "16-b": {"vertices": 16, "edges": 52, "polygon": [60, 90]},  # 16-vertex
            "16-c": {"vertices": 16, "aet": [{'9': {'3': 6}}, {'3': {'3': 4}}, {'3': {'3': 3}}, {'1': {'3': 9}}]},  
            "16-d": {"vertices": 16, "aet": [{'6': {'3': 4, '5': 1}}, {'3': {'3': 6}}, {'3': {'3': 4}}, {'3': {'3': 1, '5': 2}}, {'1': {'5': 3}}]},  
            "17-a": {"vertices": 17, "aet": [{'10': {'3': 5}}, {'4': {'3': 4, '4': 1}}, {'3': {'3': 6}}]},  
            "17-b": {"vertices": 17, "aet": [{'8': {'3': 4, '4': 1}}, {'4': {'3': 5}}, {'4': {'3': 2, '4': 2}}, {'1': {'4': 4}}]},  
            "17-d": {"vertices": 17, "aet": [{'12': {'3': 5}}, {'5': {'3': 6}}]},  # 7-capped pentagonal prism
            "17-e": {"vertices": 17, "aet": [{'6': {'3': 6}}, {'4': {'3': 3}}, {'3': {'3': 7}}, {'3': {'3': 4}}, {'1': {'3': 9}}]},
            "17-f": {"vertices": 17, "aet": [{'12': {'3': 4, '4': 1}}, {'3': {'3': 4}}, {'2': {'3': 6}}]},
            
            "18-a": {"vertices": 18, "aet": [{'12': {'3': 6}}, {'5': {'3': 4}}]},  # !8-equatorial capped pentagonal prism
            "18-c": {"vertices": 18, "aet": [{'12': {'3': 5}}, {'6': {'3': 6}}]},  # pseudo Frank-Kasper
            "18-d": {"vertices": 18, "aet": [{'12': {'3': 2, '4': 1, '6': 1}}, {'6': {'3': 2, '4': 2}}]},  # 6-capped hexagonal prism
            "18-e": {"vertices": 18, "aet": [{'6': {'3': 5}}, {'6': {'3': 3, '4': 2}}, {'6': {'3': 2, '4': 2}}]},  # double-bicapped heptagonal prism
            "19-b": {"vertices": 19, "aet": [{'15': {'3': 5}}, {'3': {'3': 7}}, {'1': {'3': 6}}]},  # distorted pseudo Frank-Kasper
            "19-c": {"vertices": 19, "aet": [{'9': {'3': 4, '4': 1}}, {'3': {'3': 5, '4': 1}}, {'3': {'3': 5}}, {'3': {'3': 4}}, {'1': {'3': 6}}]},  # pseudo Frank-Kasper
            "20-a": {"vertices": 20, "aet": [{'12': {'3': 5}}, {'8': {'3': 8}}]},
            # "20-h": {"vertices": 20, "edges": 90, "polygon": [108]},  # 20-pentagonal faced polyhedron
            "21-a": {"vertices": 21, "aet": [{'4': {'3': 3, '4': 2}}, {'4': {'3': 2, '4': 2}}, {'3': {'3': 5}}, {'2': {'3': 5, '4': 1}}, {'2': {'3': 4, '4': 1}}, {'2': {'3': 3, '4': 1}}, {'1': {'3': 6, '4': 1}},
                                             {'1': {'3': 4, '4': 2}}, {'1': {'3': 2, '4': 1}},  {'1': {'3': 1, '4': 2}}]}, # pseudo Frank-Kasper
            "22-a": {"vertices": 22, "aet": [{'16': {'3': 2, '4': 2}}, {'4': {'3': 4, '4': 2}}, {'2': {'4': 1}}]},   # polar 8-equatorial capped hexagonal prism
            "23-a": {"vertices": 23, "aet": [{'14': {'3': 6}}, {'6': {'3': 5}}, {'3': {'3': 4}}]},  # pseudo Frank-Kasper
            # "24-a": {"vertices": 24, "edges": 72, "polygon": [90, 120]},  # 6-square; 8-hexagonal faced polyhedron
            # "24-b": {"vertices": 24, "edges": 72, "polygon": [60, 90]},  # pseudo Frank-Kasper
            # "24-c": {"vertices": 24, "edges": 36, "polygon": [90]},  # truncated cube; dice
            # "24-d": {"vertices": 24, "edges": 72, "polygon": [90, 120]},  # triangle-square-hexagon faced polyhedron
            # "24-g": {"vertices": 24, "edges": 37, "polygon": [90]},  # triangle-square faced polyhedron
            # "26-a": {"vertices": 26, "edges": 78, "polygon": [60, 90]},  # $ pseudo Frank-Kasper
            # "28-a": {"vertices": 28, "edges": 84, "polygon": [60, 90]},  # $ pseudo Frank-Kasper
            "30-a": {"vertices": 30, "aet": [{'18': {'3': 4, '4': 1}}, {'6': {'3': 5, '4': 1}}, {'6': {'3': 5}}]}, 
            # "32-a": {"vertices": 32, "edges": 96, "polygon": [60, 90]},  # $ 32-vertex
        }