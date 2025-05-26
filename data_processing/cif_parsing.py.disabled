from Bio.PDB import MMCIFParser
import numpy as np
import random
import os
import re

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
    
    def get_features(self, complex_structure, encoding_type="heuristic", pdb_path=None):
        if encoding_type == "heuristic":
            return self.get_features_by_heuristic(complex_structure)
        elif encoding_type == "gold":
            if pdb_path is None:
                structure_id = complex_structure.id
                # 自动判断目录
                for base in ["data/CASF-2016/coreset", "data/refined-set"]:
                    candidate = os.path.join(base, structure_id, f"{structure_id}_protein.pdb")
                    if os.path.exists(candidate):
                        pdb_path = candidate
                        break
                if pdb_path is None:
                    raise FileNotFoundError(f"Cannot find pdb for {structure_id}")
            return self.get_features_by_reading_pdb(complex_structure, pdb_path)
        
    def get_features_by_reading_pdb(self, complex_structure, pdb_path):
        """
        从PDB和MOL2文件中提取金标准杂化状态特征
        """
        
        # 路径校验与MOL2文件搜索
        base_dir = os.path.dirname(pdb_path)
        structure_id = os.path.basename(base_dir)
        
        # 支持多种命名格式的MOL2文件搜索
        def find_mol2_file(patterns):
            for pattern in patterns:
                path = os.path.join(base_dir, f"{structure_id}{pattern}")
                if os.path.exists(path):
                    return path
                path = os.path.join(base_dir, pattern)
                if os.path.exists(path):
                    return path
            return None

        protein_mol2 = find_mol2_file(["_protein.mol2", "_pocket.mol2", "protein.mol2"])
        ligand_mol2 = find_mol2_file(["_ligand.mol2", "ligand.mol2", "_ligand_opt.mol2"])
        
        if not protein_mol2 or not ligand_mol2:
            raise FileNotFoundError(f"Required MOL2 files not found in {base_dir}")

        # MOL2解析逻辑
        def parse_mol2_hybrid(mol2_path):
            hybrid_info = {}
            try:
                with open(mol2_path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                    
                in_atom_section = False
                for line in lines:
                    if line.startswith('@<TRIPOS>ATOM'):
                        in_atom_section = True
                        continue
                    if line.startswith('@<TRIPOS>'):
                        in_atom_section = False
                        continue
                    if not in_atom_section:
                        continue

                    parts = re.split(r'\s+', line, maxsplit=7)
                    if len(parts) < 6:
                        continue
                    
                    atom_name = parts[1].strip()
                    atom_type = parts[5].strip()
                    res_name = parts[6].strip() if len(parts)>6 else "UNK"
                    
                    hybrid_info[(atom_name, res_name)] = atom_type
            except Exception as e:
                print(f"Error parsing {mol2_path}: {str(e)}")
            return hybrid_info

        # 杂化类型转换
        def type_to_hyb(atom_type):
            if not atom_type: return None
            atom_type = atom_type.lower()
            if '.1' in atom_type: return 1
            if '.2' in atom_type or '.ar' in atom_type: return 2
            if '.3' in atom_type: return 3
            return 3 if atom_type.startswith(('c','n','s')) else None

        # 主处理逻辑
        protein_hybrid = parse_mol2_hybrid(protein_mol2)
        ligand_hybrid = parse_mol2_hybrid(ligand_mol2)

        coords_protein = []
        features_protein = []
        coords_ligand = []
        features_ligand = []
        b_factors_as_plddts = []
        
        for model in complex_structure:
            for chain in model:
                for residue in chain:
                    residue_hetfield = residue.id[0]
                    if residue_hetfield == " ":  # Protein
                        molprotein = 1
                        current_hybrid = protein_hybrid
                        res_name = residue.get_resname().strip()
                    elif residue_hetfield.startswith('H_') and residue_hetfield != 'W':  # Ligand
                        molprotein = 0
                        current_hybrid = ligand_hybrid
                        res_name = residue_hetfield[2:].strip()
                    else:
                        raise ValueError(f"Unknown residue type: {residue_hetfield}")

                    for atom in residue:
                        try:
                            element = atom.element.strip().upper()
                            atomicnum = self.element_to_atomic_num.get(element, None)
                            atom_name = atom.get_name().strip()
                            
                            # 获取杂化状态
                            atom_type = current_hybrid.get((atom_name, res_name), None)
                            hyb = type_to_hyb(atom_type) if atom_type else None
                            
                            # 回退到启发式规则
                            if hyb is None and atomicnum in [6,7,16]:
                                hyb = _determine_hybridization_heu(atom, residue)
                            
                            # 特征编码
                            if hyb is not None and atomicnum in [6,7,16]:
                                feature_input = (atomicnum, hyb)
                            else:
                                feature_input = atomicnum
                                
                            feature = self.encode(feature_input, molprotein)

                            # 存储特征
                            if molprotein == 1:
                                coords_protein.append(atom.coord)
                                features_protein.append(feature)
                            else:
                                coords_ligand.append(atom.coord)
                                features_ligand.append(feature)
                                
                            b_factors_as_plddts.append(atom.get_bfactor())
                            
                        except Exception as e:
                            print(f"Skipping atom {atom.get_id()}: {str(e)}")
                            continue

        return (
            np.array(coords_protein, dtype=np.float32),
            np.array(features_protein, dtype=np.float32),
            np.array(coords_ligand, dtype=np.float32),
            np.array(features_ligand, dtype=np.float32),
            np.array(b_factors_as_plddts, dtype=np.float32)
        )
    
    def get_features_by_heuristic(self, complex_structure):
        coords_protein = []
        features_protein = []
        coords_ligand = []
        features_ligand = []
        b_factors_as_plddts = []
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
                        b_factors_as_plddts.append(atom.get_bfactor())
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
        return coords_protein, features_protein, coords_ligand, features_ligand, b_factors_as_plddts
    
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