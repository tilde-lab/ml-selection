from random import randrange
from ml_selection.data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS
from os import listdir
from os.path import isfile, join
from ml_selection.data_massage.polyhedra.search_poly import search_poly_by_entry


def get_random_s_entry() -> str:
    """Return a random existing entry from MPDS"""
    range_d = [['250138', '250943'], ['251736', '252180'], ['260286', '260628'], 
               ['261106', '261443'], ['261484', '262017'], ['300780', '301002'], 
               ['301193', '301568'], ['301631', '301930'], ['303020', '303266'], 
               ['307005', '307268'], ['309409', '309633'], ['311035', '311269'], 
               ['312010', '312365'], ['375674', '375883'], ['450086', '450366'], 
               ['450600', '450866'], ['452046', '452316'], ['453572', '453786'], 
               ['454093', '454297'], ['456222', '456470'], ['457382', '457586'], 
               ['458777', '459511'], ['460588', '460792'], ['525193', '525686'], 
               ['525871', '526232'], ['527991', '528309'], ['528583', '528811'], 
               ['532753', '532977'], ['533192', '533805'], ['534048', '534277'], 
               ['534809', '535056'], ['535575', '536024'], ['537137', '537600'], 
               ['537699', '537963'], ['539488', '539755'], ['547630', '547836'], 
               ['554011', '554306'], ['560854', '561182'], ['1000379', '1000578'],
               ['1000844', '1001062'], ['1002543', '1002749'], ['1005413', '1005761'], 
               ['1005763', '1006151'], ['1006532', '1006778'], ['1007009', '1007529'], 
               ['1007720', '1007942'], ['1007944', '1008331'], ['1008807', '1009127'], 
               ['1009199', '1009403'], ['1009671', '1009918'], ['1009920', '1010197'], 
               ['1011240', '1011477'], ['1011502', '1011918'], ['1012946', '1013182'], 
               ['1013345', '1013689'], ['1013694', '1013983'], ['1013985', '1014212'], 
               ['1021020', '1021348'], ['1022594', '1022866'], ['1022871', '1023090'], 
               ['1023172', '1023507'], ['1023989', '1024205'], ['1024207', '1024432'], 
               ['1030545', '1031336'], ['1043056', '1043317'], ['1045499', '1045699'], 
               ['1046567', '1046899'], ['1047120', '1047339'], ['1100190', '1100393'], 
               ['1100399', '1100712'], ['1102725', '1103000'], ['1120611', '1120873'], 
               ['1121038', '1121325'], ['1121633', '1121929'], ['1121967', '1122330'], 
               ['1122967', '1123403'], ['1124734', '1125111'], ['1125151', '1125487'], 
               ['1125489', '1126017'], ['1126834', '1127381'], ['1140001', '1140391'], 
               ['1141337', '1141551'], ['1142449', '1142700'], ['1142851', '1143053'], 
               ['1144231', '1144483'], ['1146331', '1146843'], ['1147867', '1148134'], 
               ['1210001', '1210247'], ['1211167', '1211636'], ['1215421', '1215789'], 
               ['1216637', '1216842'], ['1216884', '1217150'], ['1217880', '1218217'], 
               ['1218759', '1219235'], ['1221912', '1222475'], ['1222477', '1222938'], 
               ['1222940', '1223703'], ['1223706', '1223917'], ['1223919', '1224522'], 
               ['1224526', '1224870'], ['1225058', '1225450'], ['1225901', '1226180'], 
               ['1226588', '1227469'], ['1227471', '1227951'], ['1228920', '1229306'], 
               ['1229358', '1229708'], ['1230102', '1230307'], ['1231349', '1231640'], 
               ['1231705', '1232079'], ['1232081', '1232344'], ['1232476', '1233041'], 
               ['1233385', '1234367'], ['1234494', '1234822'], ['1236145', '1236811'], 
               ['1237123', '1237338'], ['1237814', '1238036'], ['1238532', '1238742'], 
               ['1239063', '1239265'], ['1239441', '1239709'], ['1240222', '1240434'], 
               ['1242119', '1242355'], ['1243188', '1243480'], ['1243526', '1243860'], 
               ['1244870', '1245106'], ['1245346', '1245883'], ['1245957', '1246189'], 
               ['1246251', '1246560'], ['1250299', '1250499'], ['1250501', '1251164'], 
               ['1251725', '1252168'], ['1300864', '1301213'], ['1321211', '1321739'], 
               ['1322063', '1322303'], ['1322747', '1323008'], ['1323010', '1323393'], 
               ['1323871', '1324228'], ['1324428', '1324650'], ['1330001', '1330229'], 
               ['1401512', '1401756'], ['1404116', '1404333'], ['1405488', '1405688'], 
               ['1406035', '1406492'], ['1406505', '1406718'], ['1406801', '1407006'], 
               ['1407333', '1407576'], ['1410629', '1410866'], ['1414685', '1414901'], 
               ['1420501', '1420776'], ['1500144', '1500390'], ['1501654', '1501893'], 
               ['1501942', '1502166'], ['1503835', '1504161'], ['1520671', '1520870'], 
               ['1521877', '1522134'], ['1532399', '1532667'], ['1532956', '1533185'], 
               ['1533513', '1533752'], ['1534241', '1534520'], ['1534686', '1534926'], 
               ['1535109', '1535318'], ['1601093', '1601324'], ['1601462', '1601685'], 
               ['1602217', '1602503'], ['1608207', '1608419'], ['1612534', '1612756'], 
               ['1613206', '1613422'], ['1613511', '1614101'], ['1614772', '1615031'], 
               ['1615141', '1615358'], ['1615463', '1615692'], ['1617609', '1617811'], 
               ['1620520', '1620762'], ['1621294', '1621561'], ['1628935', '1629143'], 
               ['1629710', '1629945'], ['1630425', '1630707'], ['1631594', '1631810'], 
               ['1632001', '1632252'], ['1632588', '1632787'], ['1632789', '1633000'], 
               ['1633743', '1633968'], ['1634608', '1634855'], ['1635698', '1635921'], 
               ['1637895', '1638103'], ['1638590', '1639340'], ['1639613', '1639865'], 
               ['1640016', '1640497'], ['1640621', '1641132'], ['1642059', '1642379'], 
               ['1642707', '1642930'], ['1645136', '1645424'], ['1701433', '1701768'], 
               ['1702118', '1702344'], ['1702470', '1702775'], ['1702777', '1703067'], 
               ['1703071', '1703631'], ['1703841', '1704083'], ['1704085', '1704328'], 
               ['1704361', '1704705'], ['1704707', '1704961'], ['1705517', '1706069'], 
               ['1706071', '1706424'], ['1707150', '1707530'], ['1707658', '1707906'], 
               ['1707996', '1708411'], ['1708721', '1708939'], ['1709709', '1710545'], 
               ['1710547', '1710928'], ['1711116', '1711388'], ['1711431', '1711675'], 
               ['1711680', '1712231'], ['1712234', '1712499'], ['1712501', '1712814'], 
               ['1713135', '1713462'], ['1713464', '1713758'], ['1713768', '1714287'], 
               ['1715490', '1715718'], ['1716442', '1716935'], ['1716940', '1717413'], 
               ['1717934', '1718206'], ['1718209', '1718632'], ['1719087', '1719381'], 
               ['1720658', '1720864'], ['1721014', '1721581'], ['1722026', '1722448'], 
               ['1722699', '1722929'], ['1723106', '1723322'], ['1723732', '1723984'], 
               ['1725503', '1725854'], ['1726102', '1726439'], ['1726441', '1726778'], 
               ['1726905', '1727121'], ['1727494', '1727703'], ['1800244', '1800469'], 
               ['1801605', '1801899'], ['1803509', '1803805'], ['1804032', '1804245'], 
               ['1804741', '1804945'], ['1810436', '1810682'], ['1810857', '1811063'], 
               ['1811470', '1811710'], ['1812935', '1813250'], ['1814439', '1814639'], 
               ['1819190', '1819716'], ['1820735', '1820935'], ['1822248', '1822584'], 
               ['1822586', '1822924'], ['1823098', '1823312'], ['1823452', '1823696'], 
               ['1823698', '1823945'], ['1823983', '1824236'], ['1824860', '1825111'], 
               ['1825337', '1825588'], ['1825624', '1825833'], ['1826736', '1827016'], 
               ['1827677', '1828002'], ['1828004', '1828284'], ['1828754', '1828995'], 
               ['1830340', '1830539'], ['1831048', '1831403'], ['1831910', '1832243'],
               ['1832338', '1832664'], ['1833937', '1834139'], ['1835278', '1835679'], 
               ['1835781', '1835981'], ['1836083', '1836290'], ['1836845', '1837176'], 
               ['1837428', '1838042'], ['1838215', '1838600'], ['1902924', '1903158'], 
               ['1903994', '1904241'], ['1906665', '1906890'], ['1907041', '1907242'], 
               ['1907469', '1907718'], ['1907735', '1907947'], ['1908000', '1908240'], 
               ['1910372', '1910729'], ['1921119', '1921509'], ['1923430', '1923687'], 
               ['1923690', '1923968'], ['1924365', '1924606'], ['1925616', '1925884'], 
               ['1927925', '1928146'], ['1928349', '1928621'], ['1928623', '1929088'], 
               ['1929472', '1929780'], ['1930544', '1930859'], ['1931010', '1931285'], 
               ['1931649', '1931901'], ['1933646', '1934113'], ['1934398', '1934654'], 
               ['1934995', '1935588'], ['1935869', '1936084'], ['1936319', '1936835'], 
               ['1936837', '1937195'], ['1937554', '1937953'], ['1937955', '1938204'], 
               ['1938380', '1938642'], ['1938658', '1938909'], ['1939313', '1939592'], 
               ['1939595', '1939794'], ['1940796', '1941182'], ['1941580', '1942064'], 
               ['1942486', '1942710'], ['1944344', '1944546'], ['1944909', '1945215'], 
               ['1946112', '1946873'], ['1947074', '1947352'], ['1947587', '1947799'], 
               ['1948594', '1948809'], ['1949972', '1950174'], ['1950416', '1950621'], 
               ['1950967', '1951210'], ['1951397', '1951775'], ['1951863', '1952153'], 
               ['1952915', '1953274'], ['1953998', '1954286'], ['1955316', '1955568'], 
               ['1957088', '1957441'], ['1040001', '1040320'], ['1048312', '1048571'], 
               ['1049510', '1049776'], ['1247663', '1247866'], ['1247869', '1248347'], 
               ['1248350', '1248582'], ['1340028', '1340371'], ['1340551', '1340834'], 
               ['1431524', '1431756'], ['1536021', '1536224'], ['1536966', '1537175'], 
               ['1728571', '1728854'], ['1957786', '1958350'], ['1958392', '1958748'], 
               ['1959100', '1959311'], ['1050343', '1050543'], ['1050572', '1050831'], 
               ['1051413', '1051892'], ['1051970', '1052169'], ['1052171', '1052503'], 
               ['1053351', '1053713'], ['1249179', '1249472'], ['1342334', '1342587'], 
               ['1538033', '1538362'], ['1729710', '1730075'], ['1730094', '1730397'], 
               ['1731036', '1731413'], ['1838796', '1839137'], ['1839179', '1839450'], 
               ['1961589', '1961807'], ['1962317', '1962611'], ['1962902', '1963195'], 
               ['1964180', '1964443'], ['1964445', '1964787'], ['1964888', '1965234']]

    entrys = range_d[randrange(len(range_d))]
    entry = randrange(int(entrys[0]), int(entrys[1]))
    return 'S' + str(entry)
    

def get_structure_with_exist_poly(sid: str, api_key: str, from_dir: bool = False, entry: str = None) -> tuple:
    """Return random entry with CIF structure and polyhedron
    
    Parameters
    ----------
    sid : str
        Sid from MPDS account
        
    Returns
    -------
    (poly : list, cif : str)
        poly - entry, polyhedrons;  cif - crystal structure
    """
    succes = False
    
    if not from_dir:
        while not(succes):
            entry = get_random_s_entry()
            
            poly = RequestMPDS.request_polyhedra(sid, [entry])
            
            if poly == []:
                continue
            elif len(poly) > 0:
                if poly[0][1] == []:
                    continue
                else:
                    cif = RequestMPDS.request_cif(api_key, entry.replace(' ', ''))
                    if str(cif) != '{"error":"Unknown entry type"}' and cif != None:
                        succes = True
            else:
                cif = RequestMPDS.request_cif(api_key, entry.replace(' ', ''))
                if str(cif) != '{"error":"Unknown entry type"}' and cif != None:
                    succes = True
    # get cif from local dir, by entry
    else:
        onlyfiles = [
            f for f in listdir("ml_selection/cif") if isfile(join("ml_selection/cif", f))
        ]
        for file_name in onlyfiles:
            if entry in file_name:
                cif = open(join("ml_selection/cif", file_name), "r").read()
                
        poly = search_poly_by_entry(entry)

    return (poly, str(cif))

if __name__ == "__main__":
    poly, cif = get_structure_with_exist_poly('SID')

        
    
                 
            
        
    
    
    