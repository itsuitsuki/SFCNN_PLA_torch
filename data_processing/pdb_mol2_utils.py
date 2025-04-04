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
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("SEQRES"):
                parts = line.split()
                # parts[2] is the chain ID and the subsequent parts are residues
                chain = parts[2]
                residues = parts[4:]
                for residue in residues:
                    aa = three_to_one.get(residue.upper(), "X")
                    chain_sequences.setdefault(chain, "")
                    chain_sequences[chain] += aa
    return chain_sequences

def extract_smiles_from_mol2(mol2_file):
    with open(mol2_file, 'r') as f:
        for line in f:
            if line.startswith("SMILES:"):
                # Return the substring after the colon, stripping whitespace from both ends
                return line.partition(":")[2].strip()
    return None


if __name__ == '__main__':
    pdb_path = "data/CASF-2016/coreset/1a30/1a30_protein.pdb"
    mol2_path = "data/CASF-2016/coreset/1a30/1a30_ligand.mol2"
    protein_sequences = extract_protein_sequence_from_pdb(pdb_path)
    smiles = extract_smiles_from_mol2(mol2_path)
    print("Protein sequences:", protein_sequences)
    print("SMILES:", smiles)
