def extract_protein_sequence_from_pdb(pdb_file):
    """Extract protein sequences from a pdb file, returning a dictionary keyed by chain ID."""
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        # Optional non-standard amino acids:
        'SEC': 'U',  # Selenocysteine
        'PYL': 'O'   # Pyrrolysine
    }
    chain_sequences = {}
    seqres_found = False
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("SEQRES"):
                seqres_found = True
                parts = line.split()
                chain = parts[2]
                residues = parts[4:]
                if chain not in chain_sequences:
                    chain_sequences[chain] = []
                for residue in residues:
                    aa = three_to_one.get(residue.upper(), "X")
                    chain_sequences[chain].append(aa)
    # 如果没有SEQRES，尝试用ATOM行提取
    if not seqres_found:
        seen = set()
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    resname = line[17:20].strip()
                    chain = line[21].strip()
                    resseq = line[22:26].strip()
                    key = (chain, resseq)
                    if key not in seen:
                        seen.add(key)
                        aa = three_to_one.get(resname.upper(), "X")
                        if chain not in chain_sequences:
                            chain_sequences[chain] = []
                        chain_sequences[chain].append(aa)
    # 转成字符串
    for chain in chain_sequences:
        chain_sequences[chain] = ''.join(chain_sequences[chain])
    return chain_sequences

def extract_smiles_from_mol2(mol2_file):
    from rdkit import Chem
    mol = Chem.MolFromMol2File(mol2_file)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract protein sequences and SMILES from PDB and MOL2 files.")
    parser.add_argument("--name", type=str, default="1a30", help="Name of the complex")
    args = parser.parse_args()
    name = args.name
    pdb_path = f"data/CASF-2016/coreset/{name}/{name}_protein.pdb"
    mol2_path = f"data/CASF-2016/coreset/{name}/{name}_ligand_opt.mol2"
    protein_sequences = extract_protein_sequence_from_pdb(pdb_path)
    smiles = extract_smiles_from_mol2(mol2_path)
    print("Protein sequences:", protein_sequences)
    print("SMILES:", smiles)
