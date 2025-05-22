from Bio.PDB import MMCIFParser
import numpy as np
import random

# def cif_parsing_demo(name="1a30"):
#     cplx_path = f"../data/complexes_16/{name}/pred.rank_0.cif"
#     parser = MMCIFParser()
#     structure = parser.get_structure(f"{name}", cplx_path)

#     for model in structure:
#         print(f"Model ID: {model.id}")
#         for chain in model:
#             print(f"Chain ID: {chain.id}")
#             for residue in chain:
#                 residue_hetfield = residue.id[0]
#                 if residue_hetfield == " ":
#                     # protein
#                     print(f"Protein")
#                     for atom in residue:
#                         print(f"Atom ID: {atom.get_id()}, Element: {atom.element}, Coordinate: {atom.coord}")
#                 elif residue_hetfield.startswith('H_') and residue_hetfield != 'W':
#                     # ligand
#                     print(f"Ligand, Residue ID: {residue.id}")
#                     for atom in residue:
#                         print(f"Atom ID: {atom.get_id()}, Element: {atom.element}, Coordinate: {atom.coord}")
#                 else:
#                     raise ValueError(f"Unknown residue hetfield: {residue_hetfield} in {name}")
                
# cif_parsing_demo()

def _determine_hybridization_heu(biopython_atom, parent_residue):
    """
    Get the hybridization state of the atom based on its element and context. (Heuristic rules)
    """
    element = biopython_atom.element.strip().upper()
    atom_name = biopython_atom.get_name() # 例如 "CA", "CB", "C", "N", "SG"
    residue_name = parent_residue.get_resname() # 例如 "ALA", "LYS", "CYS"
    is_hetatm = parent_residue.id[0].startswith('H_') # 判断是否是配体或非标准残基

    # 仅处理蛋白质中的 C, N, S (非 HETATM)
    if not is_hetatm:
        if element == 'C':
            # 主链羰基碳 "C" 通常是 sp2
            if atom_name == 'C':
                return 2 # sp2
            # 芳香环中的碳 (如 PHE, TYR, TRP, HIS 的侧链) 通常是 sp2
            elif residue_name in ["PHE", "TYR", "TRP"] and atom_name in ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]:
                return 2 # sp2
            elif residue_name == "HIS" and atom_name in ["CG", "CD2", "CE1"]: # 组氨酸咪唑环碳
                return 2 # sp2
            # 其他大部分碳 (如 CA, CB, 脂肪链侧链碳) 通常是 sp3
            else:
                return 3 # sp3

        elif element == 'N':
            # 主链肽键氮 "N" 由于共振效应，通常认为是 sp2
            if atom_name == 'N':
                return 2 # sp2
            # 组氨酸、色氨酸等环内的氮可能是 sp2
            elif residue_name == "HIS" and atom_name in ["ND1", "NE2"]:
                return 2 # sp2
            elif residue_name == "TRP" and atom_name == "NE1":
                return 2 # sp2
            # 精氨酸胍基中的 N (NE, NH1, NH2) 是 sp2
            elif residue_name == "ARG" and atom_name in ["NE", "NH1", "NH2"]:
                return 2 # sp2
            # 赖氨酸的 NZ 通常是 sp3
            elif residue_name == "LYS" and atom_name == "NZ":
                return 3 # sp3
            # 其他情况，如 ASN, GLN 侧链的 ND2, NE2 酰胺氮也是 sp2
            elif residue_name in ["ASN", "GLN"] and atom_name in ["ND2", "NE2"]:
                return 2 # sp2
            else: # 默认 sp3 (可能需要更细致的规则)
                return 3 # sp3

        elif element == 'S':
            # 半胱氨酸的 SG 通常是 sp3
            # 甲硫氨酸的 SD 通常是 sp3
            return 3 # sp3
    
    # 对于配体 (HETATM)，原子名称的意义依赖于具体的配体。
    # 这里简单返回 None，表示需要其他方法（如 RDKit）来判断配体的杂化状态。
    # 或者你可以为已知的配体类型添加特定规则。
    elif is_hetatm:
        # print(f"配体原子 {residue_name}-{atom_name} ({element}) 的杂化状态需要专门处理。")
        # 示例：如果你知道某个配体 LIG 的 C1 原子是 sp2
        # if residue_name == "LIG" and atom_name == "C1" and element == "C":
        #     return 2
        if element == 'C': return 3 # 配体C的默认值
        if element == 'N': return 3 # 配体N的默认值
        if element == 'S': return 3 # 配体S的默认值

    return None # 对于其他元素或无法确定的情况


class FeatureExtractorCIF():
    def __init__(self):
        self.atom_codes = {}
        # 'others' includs metal atoms and B atom. There are no B atoms on training and test sets. 
        others = ([3,4,5,11,12,13]+list(range(19,32))+list(range(37,51))+list(range(55,84)))
        # C and N atoms can be hybridized in three ways and S atom can be hybridized in two ways here. 
        # Hydrogen atom is also considered for feature extraction.
        
        # 1 is H, 6 is C, 7 is N, 8 is O, 9 is F, 15 is P, 16 is S, 17 is Cl, 19-32 are Arsenic to Selenium, 37-51 are Rubidium to Indium, 55-84 are Cesium to Polonium
        atom_types = [1,(6,1),(6,2),(6,3),(7,1),(7,2),(7,3),8,15,(16,2),(16,3),34,[9,17,35,53],others] 
        # (6,1): C atom with sp hybridization, (6,2) C sp2, (6,3) C sp3
        # (16,2): S sp2, (16,3) S sp3 sp3d
        
        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
                
            else:
                self.atom_codes[j] = i              
        
        self.sum_atom_types = len(atom_types)
        
        self.element_to_atomic_num = {
            'H': 1, 'HE': 2,
            'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
            'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
            'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26,
            'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34,
            'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42,
            'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50,
            'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58,
            'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66,
            'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 'TA': 73, 'W': 74,
            'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82,
            'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86
        }
        
    # Onehot encoding of each atom. The atoms in protein or ligand are treated separately.
    def encode(self, atomic_num, molprotein):
        encoding = np.zeros(self.sum_atom_types*2)
        if molprotein == 1: # protein
            encoding[self.atom_codes[atomic_num]] = 1.0 
        else: # ligand
            encoding[self.sum_atom_types+self.atom_codes[atomic_num]] = 1.0
        
        return encoding
    
    def get_features(self, complex_structure, encoding_type="heuristic"):
        if encoding_type == "heuristic":
            return self.get_features_by_heuristic(complex_structure)
        elif encoding_type == "gold":
            return self.get_features_by_reading_pdb(complex_structure, complex_structure)
        
    def get_features_by_reading_pdb(self, complex_structure, pdb_path):
        # read the hybridization from original pdb
        # TODO
        raise NotImplementedError("This method is not implemented yet.")
    
    def get_features_by_heuristic(self, complex_structure):
        coords_protein = []
        features_protein = []
        coords_ligand = []
        features_ligand = []
        for model in complex_structure:
            for chain in model:
                
                for residue in chain:
                    residue_hetfield = residue.id[0]
                    if residue_hetfield == " ": # protein
                        molprotein = 1
                    elif residue_hetfield.startswith('H_') and residue_hetfield != 'W': # ligand
                        molprotein = 0
                    else:
                        raise ValueError(f"Unknown residue hetfield: {residue_hetfield} in {complex_structure.id}")
                    for atom in residue:
                        # print(f"Atom ID: {atom.get_id()}, Element: {atom.element}, Coordinate: {atom.coord}")
                        element = atom.element
                        atomicnum = self.element_to_atomic_num.get(element, None)
                        
                        if atomicnum in [6,7,16]:
                            # 处理 C, N, S 原子
                            hybridization = _determine_hybridization_heu(atom, residue)
                            if hybridization is not None:
                                if molprotein == 1:
                                    features_protein.append(self.encode((atomicnum, hybridization), molprotein))
                                else:
                                    features_ligand.append(self.encode((atomicnum, hybridization), molprotein))
                            else:
                                raise ValueError(f"Hybridization is None when processing {atom.get_id()}") 
                        else:
                            # 处理其他原子
                            if molprotein == 1:
                                features_protein.append(self.encode(atomicnum, molprotein))
                            else:
                                features_ligand.append(self.encode(atomicnum, molprotein))
                        # id = atom.get_id()
                        coord = atom.coord
                        if molprotein == 1:
                            coords_protein.append(coord)
                        else:
                            coords_ligand.append(coord)
        coords_protein = np.array(coords_protein, dtype=np.float32) # shape [num_atoms, 3]
        features_protein = np.array(features_protein, dtype=np.float32) # shape [num_atoms, 28]
        coords_ligand = np.array(coords_ligand, dtype=np.float32) # shape [num_atoms, 3]
        features_ligand = np.array(features_ligand, dtype=np.float32) # shape [num_atoms, 28]
        return coords_protein, features_protein, coords_ligand, features_ligand
    
    # Define the rotation matrixs of 3D stuctures.
    def rotation_matrix(self, t, roller):
        if roller==0:
            return np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]])
        elif roller==1:
            return np.array([[np.cos(t),0,-np.sin(t)],[0,1,0],[np.sin(t),0,np.cos(t)]])
        elif roller==2:
            return np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]])

    # Generate 3d grid or 4d tensor. Each grid represents a voxel. Each voxel represents the atom in it by onehot encoding of atomic type.
    # Each complex in train set is rotated 9 times for data amplification.
    # The complexes in core set are not rotated. 
    # The default resolution is 20*20*20.
    def grid(self, coords, features, resolution=1.0, max_dist=10.0, n_amplification=0):
        """
        Generate a grid representation of the protein-ligand complex.

        n_amplification: int = number of rotations to apply to the complex.
        if n_amplification=0, the complex is not rotated, only the original complex is used.
        (return shape [1, 20, 20, 20, features.shape[1]])
        otherwise, the complex is rotated n_amplification times,
        and return shape [n_amplification + 1, 20, 20, 20, features.shape[1]].
        """
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  

        grid=np.zeros((n_amplification + 1, 20, 20, 20, features.shape[1]),dtype=np.float32)
        x=y=z=np.array(range(-10, 10),dtype=np.float32)+0.5
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z) 
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[0,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
                
                
        # for testing, the complex is not rotated. n_rotations=1
        
        for j in range(n_amplification):
            theta = random.uniform(np.pi/18,np.pi/2)
            roller = random.randrange(3)
            coords = np.dot(coords, self.rotation_matrix(theta,roller))
            for i in range(len(coords)):
                coord=coords[i]
                tmpx=abs(coord[0]-x)
                tmpy=abs(coord[1]-y)
                tmpz=abs(coord[2]-z)
                
                # because the grid is 20x20x20, and tmp(x,y,z) out of 19.5 means that the atom is out of the grid, then eliminate it
                if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                    grid[j+1,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
                
        return grid