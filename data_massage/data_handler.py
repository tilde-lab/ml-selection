import numpy as np
import pandas as pd
import yaml
from ase import Atoms
from ase.data import chemical_symbols
from pandas import DataFrame

from data.mendeleev_table import periodic_numbers
from data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS
# change path if another
from metis_backend.metis_backend.structures.struct_utils import \
    order_disordered
from data_massage.create_polyheadra import get_poly_elements, get_int_poly_type, size_customization


class DataHandler:
    """
    Receiving and processing data.
    Implemented support for processing data just from MPDS
    """

    def __init__(self, is_MPDS: bool, dtype: int = 1) -> None:
        """
        Initializes the client to access database

        Parameters
        ----------
        is_MPDS : bool
            Parm is True, if you want to use MPDS
        api_key : str
            Key from MPDS-account
        dtype : int
            Indicates the type of data being requested
        """
        with open("/root/projects/ml-selection/configs/config.yaml", "r") as yamlfile:
            yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.api_key = yaml_f["api_key"]
            self.polyheadra_path = yaml_f["raw_polyhedra_path"]

        if is_MPDS:
            self.client_handler = RequestMPDS(dtype=dtype, api_key=self.api_key)
        else:
            self.client_handler = None
        self.available_dtypes = [1, 4]
        self.dtype = dtype

    def just_seebeck(
        self, max_value: int, min_value: int, is_uniq_phase_id: bool = False
    ) -> DataFrame:
        """
        Get Seebeck coefficient from db

        Parameters
        ----------
        max_value : int
            Max value for range required data
        min_value : int
            Min value for range required data
        is_uniq_phase_id : bool
            Affects the filtering out of data with duplicate 'phase_id',
            if parm is True data will contain only a unique 'phase_id'

        Returns
        -------
        res_dfrm : DataFrame
            Seebeck coefficients in DataFrame format
        """
        res_dfrm = pd.DataFrame(columns=["Phase", "Formula", "Seebeck coefficient"])

        for data_type in self.available_dtypes:
            self.client_handler = RequestMPDS(dtype=data_type, api_key=self.api_key)

            dfrm = self.client_handler.make_request(is_seebeck=True)

            # remove outliers in value of Seebeck coefficient
            if max_value != None:
                for index, row in dfrm.iterrows():
                    if (
                        max_value < row["Seebeck coefficient"]
                        or row["Seebeck coefficient"] < min_value
                    ):
                        dfrm.drop(index, inplace=True)

            res_dfrm = self.combine_data(dfrm, res_dfrm)

        # leave only one phase_id value
        if is_uniq_phase_id:
            res_dfrm = self.just_uniq_phase_id(res_dfrm)

        return res_dfrm

    def to_order_disordered_str(
        self, phases: list, is_uniq_phase_id: bool = True
    ) -> DataFrame:
        """
        Make order in disordered structures.
        Return pandas Dataframe with ordered structures
        """
        # get disordered structures from db, save random structure for specific 'phase_id'
        all_data_df = self.cleaning_trash_data(
            self.client_handler.make_request(is_structure=True, phases=phases),
            idx_check=5,
        )
        if is_uniq_phase_id:
            all_data_df = self.just_uniq_phase_id(all_data_df)

        disordered_str = []
        for atomic_str in all_data_df.values.tolist():
            if atomic_str and any([occ != 1 for occ in atomic_str[1]]):
                disordered_str.append(atomic_str)

        disordered_df = pd.DataFrame(
            disordered_str,
            columns=[
                "phase_id",
                "occs_noneq",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
                "entry",
                "temperature"
            ],
        )

        result_list = []
        atoms_list = []

        # create Atoms objects
        for index, row in disordered_df.iterrows():
            # info for Atoms obj
            disordered = {"disordered": {}}

            basis_noneq = row["basis_noneq"]
            els_noneq = row["els_noneq"]
            occs_noneq = row["occs_noneq"]

            for idx, (position, element, occupancy) in enumerate(
                zip(basis_noneq, els_noneq, occs_noneq)
            ):
                # Add information about disorder to dict
                disordered["disordered"][idx] = {element: occupancy}
            crystal = Atoms(
                symbols=row["els_noneq"],
                positions=row["basis_noneq"],
                cell=row["cell_abc"],
                info=disordered,
            )
            atoms_list.append(crystal)

        # make ordered structures
        for i, crystal in enumerate(atoms_list):
            obj, error = order_disordered(crystal)
            if not error:
                result_list.append(
                    [
                        disordered_df["phase_id"].tolist()[i],
                        obj.get_cell_lengths_and_angles().tolist(),
                        disordered_df["sg_n"].tolist()[i],
                        obj.get_positions().tolist(),
                        list(obj.symbols),
                        disordered_df["entry"].tolist()[i],
                        disordered_df["temperature"].tolist()[i]
                    ]
                )
            else:
                print(error)

        new_ordered_df = pd.DataFrame(
            result_list,
            columns=[
                "phase_id",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
                "entry",
                "temperature"
            ],
        )

        new_ordered_df = self.change_disord_on_ord(
            all_data_df.values.tolist(), new_ordered_df.values.tolist()
        )
        result_df = self.choose_temperature(new_ordered_df)

        return result_df

    def add_seebeck_by_phase_id(
        self, seebeck_df: DataFrame, structures_df: DataFrame
    ) -> DataFrame:
        try:
            seebeck_df = seebeck_df.rename(columns={"Phase": "phase_id"})
        except:
            pass
        dfrm = pd.merge(seebeck_df, structures_df, on="phase_id", how="inner")
        return dfrm

    def just_uniq_phase_id(self, df: DataFrame) -> DataFrame:
        """
        Save one example for a specific 'phase_id', deletes subsequent ones
        """
        try:
            mask = ~df["Phase"].duplicated()
        except:
            mask = ~df["phase_id"].duplicated()
        result_df = df[mask]
        return result_df

    def cleaning_trash_data(
        self, df: DataFrame, idx_check: int = 5, type_of_trash=[]
    ) -> DataFrame:
        """
        Delete data with wrong information or empty data
        """
        data = df.values.tolist()
        data_res = []

        for row in data:
            if row[idx_check] != type_of_trash and row[idx_check] != None:
                data_res.append(row)
            else:
                print("Removed garbage data:", row)
        data = pd.DataFrame(
            data_res,
            columns=[
                "phase_id",
                "occs_noneq",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
                "entry",
                "temperature"
            ],
        )
        return data

    def combine_data(self, data_f: DataFrame, data_s: DataFrame) -> DataFrame:
        """Simply connects 2 dataframes"""
        combined_df = pd.concat([data_f, data_s])
        return combined_df

    def change_disord_on_ord(self, data_disord: list, ordered: list) -> DataFrame:
        """
        Create DataFrame with updated ordered values for disordered data.
        Other structures copy to new list without changes

        Parameters
        ----------
        data_disord : list
            Made of 'phase_id', 'occs_noneq', 'cell_abc',
            'sg_n', 'basis_noneq', 'els_noneq', 'entry', 'temperature'
        ordered : list
            Made of next columns: 'phase_id', 'cell_abc',
            'sg_n', 'basis_noneq', 'els_noneq', 'entry', 'temperature'
        """
        update_data = []
        loss_str = 0

        for dis_sample in data_disord:
            for i, ord_sample in enumerate(ordered):
                if dis_sample[6] == ord_sample[5]:
                    update_data.append(ord_sample)
                    break
                elif i == len(ordered) - 1:
                    # check that data is really sorted
                    if not (any([occ != 1 for occ in dis_sample[1]])):
                        update_data.append(
                            [
                                dis_sample[0],
                                dis_sample[2],
                                dis_sample[3],
                                dis_sample[4],
                                dis_sample[5],
                                dis_sample[6],
                            ]
                        )
                    else:
                        # see errors occurred in 'to_order_disordered_str'
                        loss_str += 1
                        print(
                            f"Missing {loss_str} structures that could not pass ordering"
                        )

        dfrm = pd.DataFrame(
            update_data,
            columns=[
                "phase_id",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
                "entry",
                "temperature"
            ],
        )
        return dfrm

    def get_bv_descriptor(self, ase_obj, kappa: int = None, overreach: bool = False):
        """
        From ASE object obtain a vectorized atomic structure
        populated to a certain fixed relatively big volume
        defined by kappa
        """
        if not kappa:
            kappa = 18
        if overreach:
            kappa *= 2

        norms = np.array([np.linalg.norm(vec) for vec in ase_obj.get_cell()])
        multiple = np.ceil(kappa / norms).astype(int)
        ase_obj = ase_obj.repeat(multiple)
        com = ase_obj.get_center_of_mass()
        ase_obj.translate(-com)
        del ase_obj[
            [
                atom.index
                for atom in ase_obj
                if np.sqrt(np.dot(atom.position, atom.position)) > kappa
            ]
        ]

        ase_obj.center()
        ase_obj.set_pbc((False, False, False))
        sorted_seq = np.argsort(
            np.fromiter((np.sqrt(np.dot(x, x)) for x in ase_obj.positions), np.float)
        )
        ase_obj = ase_obj[sorted_seq]

        elements, positions = [], []
        for atom in ase_obj:
            elements.append(periodic_numbers[chemical_symbols.index(atom.symbol)] - 1)
            positions.append(
                int(
                    round(
                        np.sqrt(
                            atom.position[0] ** 2
                            + atom.position[1] ** 2
                            + atom.position[2] ** 2
                        )
                        * 10
                    )
                )
            )

        return np.array([elements, positions])

    def to_cut_vectors_struct(
        self, file_path: str = None, dfrm: DataFrame = None
    ) -> list:
        """
        Reduce vectors representing structure to 100 elements

        Parameters
        ----------
        file_path : str, optional
            Path to file which containing data
        dfrm : DataFrame, optional
            DataFrame, which contain processed Seebeck and structures

        Returns
        -------
        dfrm_str, dfrm_seeb
           2 DataFrames, first consist of atoms and distance, second - Seebeck value
        """
        if file_path:
            dfrm = pd.read_csv(file_path)

        try:
            d_list = dfrm.values.tolist()
        except NameError:
            print("Variable dfrm is not defined")
            return None

        objs = []
        seebeck = []

        for item in d_list:
            crystal = Atoms(symbols=item[6], positions=item[5], cell=item[3])
            vectors = self.get_bv_descriptor(crystal, kappa=40)
            if len(vectors[0]) < 100:
                continue
            elif len(vectors[0]) == 100:
                objs.append(vectors)
                seebeck.append(item[2])
            else:
                objs.append(vectors[:, :100])
                seebeck.append(item[2])

        dfrm_str = pd.DataFrame(
            [i.tolist() for i in objs], columns=["atom", "distance"]
        )
        dfrm_seeb = pd.DataFrame(seebeck, columns=["Seebeck coefficient"])

        return [dfrm_str, dfrm_seeb]

    def add_polyhedra(self, structure_path: str) -> DataFrame:
        """
        Add polyhedra info by entry for each structure entry

        Parameters
        ----------
        structure_path : str
            Path to json file with structures

        Returns
        -------
        dfrm : DataFrame
           Table with next columns:
           'phase_id', 'cell_abc', 'sg_n', 'basis_noneq',
           'els_noneq', 'entry', 'Site', 'Type', 'Composition'
        """
        poly = pd.read_csv(self.polyheadra_path).rename(columns={"Entry": "entry"})
        structs = pd.read_json(structure_path, orient="split")

        dfrm = pd.merge(structs, poly, on="entry", how="inner")
        return dfrm


    @classmethod
    def vectors_count_elements(cls, elements):
        count_el = [0 for i in range(118)]
        for el in elements:
            count_el[el-1] += 10
        return count_el


    @classmethod
    def process_polyhedra(
            cls, crystals_json_path: str, features: int = 2, is_one_hot: bool = False
    ) -> DataFrame:
        """
        Create descriptor from polyhedra

        Parameters
        ----------
        crystals_json_path : str
            Path to json file with structures
        features : int
            If 2 -> features: elements, poly (number of vertex + number of type poly)
            If 3 -> features: elements, number of vertex in poly, type of poly
            If 0 -> features: elements without size customization (data just for graph models), type of poly
        is_one_hot : bool, optional
            Present elements in vectors of count, where periodic number of element is index in vector

        Returns
        -------
        dfrm : DataFrame
           Table with next columns:
           'phase_id', 'poly_elements', 'poly_vertex', 'poly_type', 'temperature'
           or 'phase_id', 'poly_elements', 'poly_type', 'temperature'
        """
        crystals = pd.read_json(crystals_json_path, orient='split').values.tolist()
        poly_store = []
        descriptor_store = []

        if features == 3 and not(is_one_hot):
            columns = ['phase_id', 'poly_elements', 'poly_vertex', 'poly_type', 'temperature']
        else:
            columns = ['phase_id', 'poly_elements', 'poly_type', 'temperature']

        for poly in crystals:
            elements = get_poly_elements(poly)

            if elements == [None]:
                continue
            if is_one_hot:
                elements = cls.vectors_count_elements(elements)

            vertex, p_type = get_int_poly_type(poly)

            # features: elements, poly (number of vertex + number of type poly)
            if features == 2 and not(is_one_hot):
                poly_type = vertex + p_type
                poly_type_large = [poly_type] * 100

            # features: elements, number of vertex in poly, type of poly (+t if is_temperature==True)
            elif features == 3 and not(is_one_hot):
                vertex_large, p_type_large = [vertex] * 100, [p_type] * 100
                poly_type_large = [vertex_large, p_type_large]

            elif features == 0 or is_one_hot:
                poly_type_large = [p_type] * 100
            else:
                return None

            if features != 0 and not(is_one_hot):
                elements_large = size_customization(elements)

            # elements without size customization (just for graph models)
            else:
                elements_large = elements

            # replay protection
            if [elements_large, poly_type_large] not in descriptor_store:
                descriptor_store.append([elements_large, poly_type_large])
                temperature = poly[6]
                if features == 2 or features == 0 or is_one_hot:
                    poly_store.append([poly[0], elements_large, poly_type_large, temperature])
                elif features == 3:
                    poly_store.append([poly[0], elements_large, vertex_large, p_type_large, temperature])

        return pd.DataFrame(poly_store, columns=columns)

    def choose_temperature(self, dfrm: DataFrame) -> DataFrame:
        """
        Choose needed temperature from available. If there is no data, it assigns room temperature

        Parameters
        ----------
        dfrm : DataFrame
            Consist of: "phase_id", "cell_abc", "sg_n", "basis_noneq",
            "els_noneq", "entry", "temperature"

        Returns
        -------
        dfrm : DataFrame
        """
        columns = dfrm.columns.tolist()
        data_list = dfrm.values.tolist()

        for row in data_list:
            # if not available t in data, set room t
            if row[6] == None:
                row[6] = 298
            elif type(row[6]) == list:
                # get first temperature from available
                if row[6][0] != None:
                    row[6] = row[6][0]
                # get third temperature from available
                elif len(row[6]) > 2:
                    if row[6][2] != None:
                        row[6] = row[6][2]
                    else:
                        row[6] = 298
                # if not available t in data, set room t
                else:
                    row[6] = 298
            # if not available t in data, set room t
            else:
                row[6] = 298

        result_dfrm = pd.DataFrame(data_list, columns=columns)

        return result_dfrm


