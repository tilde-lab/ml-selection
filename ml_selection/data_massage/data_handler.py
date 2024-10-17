import numpy as np
import pandas as pd
import polars as pl
import yaml
from ase import Atoms
from ase.data import chemical_symbols
from polars import DataFrame

from ml_selection.structures_props.mendeleev_table import periodic_numbers
from ml_selection.data_massage.polyhedra.create_polyhedra import (
    get_poly_elements,
    size_customization
)
from ml_selection.data_massage.database_handlers.MPDS.request_to_mpds import RequestMPDS

# change path if another
from metis_backend.structures.struct_utils import order_disordered


class DataHandler:
    """
    Receiving and processing data.
    Implemented support for processing data just from MPDS
    """

    def __init__(self, is_MPDS: bool = True, dtype: int = 1) -> None:
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
        with open("ml_selection/configs/config.yaml", "r") as yamlfile:
            yaml_f = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.api_key = yaml_f["api_key"]
            self.polyheadra_path = yaml_f["raw_polyhedra_path"]

        if is_MPDS:
            self.client_handler = RequestMPDS(dtype=dtype, api_key=self.api_key)
            self.db = "MPDS"
        else:
            self.client_handler = None
            self.db = "MP"
        # ab_initio or peer_rev
        self.available_dtypes = [1]
        self.dtype = dtype

    def just_phys_prop(
        self,
        max_value: int,
        min_value: int,
        is_uniq_phase_id: bool = False,
        phys_prop: str = "Seebeck coefficient",
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
        res_dfrm = pl.DataFrame({"Phase": [], "Formula": [], phys_prop: []})

        for data_type in self.available_dtypes:
            self.client_handler = RequestMPDS(dtype=data_type, api_key=self.api_key)

            dfrm = self.client_handler.make_request(
                is_phys_prop=True, phys_prop=phys_prop
            )

            # remove outliers in value
            if max_value:
                dfrm = dfrm.filter(
                    (pl.col(phys_prop) >= min_value) & (pl.col(phys_prop) <= max_value)
                )
            if len(res_dfrm) == 0:
                res_dfrm = dfrm
            else:
                res_dfrm = self.combine_data(dfrm, res_dfrm)

        # leave only one phase_id value
        if is_uniq_phase_id:
            res_dfrm = self.just_uniq_phase_id(res_dfrm)

        return res_dfrm

    def to_order_disordered_str(
        self,
        phases: list,
        is_uniq_phase_id: bool = True,
        return_not_clean_not_ordered: bool = False,
    ) -> DataFrame:
        """
        Make order in disordered structures.
        Return polars Dataframe with ordered structures
        """
        if return_not_clean_not_ordered:
            dfrm = pl.from_pandas(
                self.client_handler.make_request(is_structure=True, phases=phases)
                .to_pandas()
                .drop("formula", axis=1)
            )
            return dfrm
        # get disordered structures from db, save random structure for specific 'phase_id'
        all_data_df = pl.from_pandas(
            self.cleaning_trash_data(
                self.client_handler.make_request(is_structure=True, phases=phases),
                idx_check=5,
            )
            .to_pandas()
            .drop_duplicates(subset=["formula", "sg_n"])
        ).drop("formula")
        if is_uniq_phase_id:
            all_data_df = self.just_uniq_phase_id(all_data_df)

        disordered_str = []
        for atomic_str in all_data_df.rows():
            if list(atomic_str) and any([occ != 1 for occ in atomic_str[1]]):
                disordered_str.append(list(atomic_str))

        disordered_df = pl.DataFrame(
            disordered_str,
            schema=[
                "phase_id",
                "occs_noneq",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
                "entry",
                "temperature",
            ],
        )

        result_list = []
        atoms_list = []

        # create Atoms objects
        for index in range(len(disordered_df)):
            # info for Atoms obj
            disordered = {"disordered": {}}

            row = disordered_df.row(index)
            basis_noneq = row[4]
            els_noneq = row[5]
            occs_noneq = row[1]
            cell_abc = row[2]

            for idx, (position, element, occupancy) in enumerate(
                zip(basis_noneq, els_noneq, occs_noneq)
            ):
                # Add information about disorder to dict
                disordered["disordered"][idx] = {element: occupancy}
            crystal = Atoms(
                symbols=els_noneq,
                positions=basis_noneq,
                cell=cell_abc,
                info=disordered,
            )
            atoms_list.append(crystal)

        # make ordered structures
        for i, crystal in enumerate(atoms_list):
            obj, error = order_disordered(crystal)
            if not error:
                result_list.append(
                    # disordered_df consist of next columns:
                    # phase_id, occs_noneq, cell_abc, sg_n, basis_non, els_noneq, entry, temperature
                    [
                        disordered_df.row(i)[0],
                        obj.get_cell_lengths_and_angles().tolist(),
                        disordered_df.row(i)[3],
                        obj.get_positions().tolist(),
                        list(obj.symbols),
                        disordered_df.row(i)[6],
                        disordered_df.row(i)[7],
                    ]
                )
            else:
                print(error)

        new_ordered_df = pl.DataFrame(
            result_list,
            schema=[
                "phase_id",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
                "entry",
                "temperature",
            ],
        )

        new_ordered_df = self.change_disord_on_ord(
            [list(all_data_df.row(i)) for i in range(len(all_data_df))],
            [list(new_ordered_df.row(i)) for i in range(len(new_ordered_df))],
        )
        result_df = self.choose_temperature(new_ordered_df)

        return result_df

    def add_phys_prop_by_phase_id(
        self, phys_prop_df: DataFrame, structures_or_sg__df: DataFrame
    ) -> DataFrame:
        if self.db == "MPDS":
            try:
                phys_prop_df = phys_prop_df.rename({"Phase": "phase_id"})
            except:
                pass
            dfrm = pd.merge(
                phys_prop_df.to_pandas(),
                structures_or_sg__df.to_pandas(),
                on="phase_id",
                how="inner",
            )
        else:
            dfrm = pd.merge(
                phys_prop_df.to_pandas(),
                structures_or_sg__df.to_pandas(),
                on="identifier",
                how="inner",
            )
        return pl.from_pandas(dfrm)

    def just_uniq_phase_id(self, df: DataFrame) -> DataFrame:
        """
        Save one example for a specific 'phase_id', deletes subsequent ones
        """
        df = df.to_pandas()
        try:
            mask = ~df["Phase"].duplicated()
        except:
            mask = ~df["phase_id"].duplicated()
        result_df = df[mask]
        return pl.from_pandas(result_df)

    def cleaning_trash_data(
        self, df: DataFrame, idx_check: int = 5, type_of_trash=[]
    ) -> DataFrame:
        """
        Delete data with wrong information or empty data
        """
        data = [list(df.row(i)) for i in range(len(df))]
        data_res = []

        for row in data:
            if row[idx_check] != type_of_trash and row[idx_check] != None:
                data_res.append(row)
            else:
                print("Removed garbage data:", row)
        data = pl.DataFrame(data_res, schema=df.columns)
        return data

    def combine_data(self, data_f: DataFrame, data_s: DataFrame) -> DataFrame:
        """Simply connects 2 dataframes"""
        combined_df = pl.concat([data_f, data_s])
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
                                dis_sample[7],
                            ]
                        )
                    else:
                        # see errors occurred in 'to_order_disordered_str'
                        loss_str += 1
                        print(
                            f"Missing {loss_str} structures that could not pass ordering"
                        )

        dfrm = pl.DataFrame(
            update_data,
            schema=[
                "phase_id",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
                "entry",
                "temperature",
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
            dfrm = pl.read_csv(file_path)

        try:
            d_list = [list(dfrm.row(i)) for i in range(len(dfrm))]
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

        dfrm_str = pl.DataFrame([i.tolist() for i in objs], schema=["atom", "distance"])
        dfrm_seeb = pl.DataFrame(seebeck, schema=["Seebeck coefficient"])

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
        poly = pl.read_csv(self.polyheadra_path).rename({"Entry": "entry"})
        structs = pl.read_json(structure_path)

        dfrm = structs.join(poly, on="entry", how="inner")
        return dfrm

    @classmethod
    def vectors_count_elements(cls, elements: list) -> list:
        """
        Create vector with count of element by index from Mendeleev table

        Parameters
        ----------
        elements : list
            Periodic numbers of elements from Mendeleev table

        Returns
        -------
        count_el : list
            Number of count for each element
        """
        count_el = [0 for i in range(118)]
        for el in elements:
            count_el[el - 1] += 10
        return count_el

    @classmethod
    def process_polyhedra(cls, crystals_json_path: str) -> DataFrame:
        """
        Create descriptor from polyhedra

        Parameters
        ----------
        crystals_json_path : str
            Path to json file with structures with next columns:
            'phase_id', 'occs_noneq', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq',
            'entry', 'temperature', 'Site', 'Type', 'Composition'

        Returns
        -------
        dfrm : DataFrame
           Table with next columns:
           "phase_id", "poly_elements", "poly_type"
        """
        crystals = pl.read_json(crystals_json_path)
        crystals = [list(crystals.row(i)) for i in range(len(crystals))]
        descriptor_store = []

        columns = ["phase_id", "poly_elements", "poly_type"]

        for poly in crystals:
            elements = get_poly_elements(poly)

            if elements == [None]:
                continue

            # features: elements, number of atoms
            poly_type = len(elements) * 100
            elements_large = size_customization(elements)

            # replay protection
            if [poly[0], elements_large, poly_type] not in descriptor_store:
                descriptor_store.append([poly[0], elements_large, poly_type])

        return pl.DataFrame(descriptor_store, schema=columns)

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
            Dataframe with the same columns
        """
        columns = dfrm.columns
        data_list = [list(dfrm.row(i)) for i in range(len(dfrm))]

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

        result_dfrm = pl.DataFrame(data_list, schema=columns)

        return result_dfrm

    def convert_mp_data_to_dataframe(self, mp_data: list) -> pl.DataFrame:
        """
        Convert answer in Materials Projects format to Polars DataFrame

        Parameters
        ----------
        mp_data : list
            Consist of any dicts with keys: 'identifier', 'data', 'formula'.
            Value by key 'data' consist of dict with keys: 'n', 'p'

        Returns
        -------
        pl.DataFrame
            Consist of columns: "identifier", "formula", "Seebeck coefficient"
        """
        identifier, seebeck, formula = [], [], []
        for row in mp_data:
            identifier.append(row["identifier"])
            seebeck.append(row["data"]["S"])
            formula.append(row["formula"])
        return pl.DataFrame(
            {
                "identifier": identifier,
                "formula": formula,
                "Seebeck coefficient": seebeck,
            },
            schema=["identifier", "formula", "Seebeck coefficient"],
        )


if __name__ == "__main__":
    with open(
        "/root/projects/ml-selection/data/mp_database/space_group_mp.json", "r"
    ) as f:
        data = f.read()
    handler = DataHandler(True)
