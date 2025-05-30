import os
import glob
import shutil
from datasets import Dataset as HFDataset
from datasets import Features, Array5D, Value, concatenate_datasets
from pdb_parsing import FeatureExtractorPDB
from cif_parsing import FeatureExtractorCIF
from Bio.PDB import MMCIFParser
import random
import openbabel as ob
from openbabel import pybel
import numpy as np
import torch
from tqdm import tqdm
import warnings

def create_real_dataset_aug():
    # stuff test set protein-ligand complexes into grids & save them
    test_grids = []
    test_labels = []
    test_names = []
    test_features = Features(
        {
            "grid": Array5D(dtype="int32", shape=(10, 28, 20, 20, 20)),
            "label": Value(dtype="float32"), # (10,)
            "name": Value(dtype="string"), # (10,)
        }
    )
    test_data_dir = "data/CASF-2016/coreset"
    test_complex_names = glob.glob(os.path.join(test_data_dir, "*"))
    test_complex_names = [os.path.basename(name) for name in test_complex_names]
    # sort
    test_complex_names.sort()
    extractor = FeatureExtractorPDB()
    test_affinity = {}
    with open("data/refined-set/index/INDEX_general_PL_data.2019", "r") as f:
        for line in f.readlines():
            if line[0] != '#' and line.split()[0] in test_complex_names:
                test_affinity[line.split()[0]] = float(line.split()[3])
    for complex_name in tqdm(test_complex_names, desc="Processing test complexes"):
        # read mol2 and pdb file
        mol2_file = os.path.join(test_data_dir, complex_name, complex_name + "_ligand.mol2")
        pdb_file = os.path.join(test_data_dir, complex_name, complex_name + "_protein.pdb")
        ligand = pybel.readfile("mol2", mol2_file).__next__() # the very first one ligand molecule
        # print(ligand)
        protein = pybel.readfile("pdb", pdb_file).__next__() # the very first one protein molecule
        coords1, features1 = extractor.get_features(protein, 1) # protein, shape 
        coords2, features2 = extractor.get_features(ligand, 0) # ligand
        center = (np.max(coords2, axis=0) + np.min(coords2, axis=0)) / 2
        coords_cat = np.concatenate((coords1, coords2), axis=0)
        
        
        features_cat = np.concatenate((features1, features2), axis=0)
        assert coords_cat.shape[0] == features_cat.shape[0], "coords and features should have the same number of atoms"
        coords_cat = coords_cat - center
        grid = extractor.grid(coords_cat, features_cat, n_amplification=9) # shape (10, 20, 20, 20, 28) # NOTE: FOR TEST TIME AUGMENTATION, N AMP = 9
        # permute 0,4,1,2,3
        # Use numpy’s transpose to permute axes (equivalent to torch.permute)
        grid = np.transpose(grid, (0, 4, 1, 2, 3))  # shape (10, 28, 20, 20, 20)
        test_grids.append(grid) # each grid is a tensor shape (10, 20, 20, 20, 28)
        test_labels.append(test_affinity[complex_name])
        test_names.append(complex_name) # each name is a string

    # values of the test_affinity dict -> lists
    test_dataset = HFDataset.from_dict({"grid": test_grids, "label": test_labels, "name": test_names}, features=test_features)
    # save the test_affinity dict -> arrow
    test_dataset.save_to_disk("data/aug/real_test_aug")
    
def create_pred_dataset_aug():
    test_data_dir = "data/CASF-2016/coreset"
    test_complex_names = glob.glob(os.path.join(test_data_dir, "*"))
    test_complex_names = [os.path.basename(name) for name in test_complex_names]
    # sort
    test_complex_names.sort()

    features = Features(
        {
            "grid": Array5D(dtype="int32", shape=(10, 28, 20, 20, 20)),
            "label": Value(dtype="float32"), # (10,)
            "name": Value(dtype="string"), # (10,)
            "pLDDT": Value(dtype="float32"), # (10,)
        }
    )

    extractor = FeatureExtractorCIF()
    test_grids = []
    test_labels = []
    test_affinity = {}
    test_names = []
    test_plddt = []
    with open("data/refined-set/index/INDEX_general_PL_data.2019", "r") as f:
        for line in f.readlines():
            if line[0] != '#' and line.split()[0] in test_complex_names:
                test_affinity[line.split()[0]] = float(line.split()[3])
    # print(test_complex_names)
    for complex_name in tqdm(test_complex_names, desc="Processing test complexes", total=len(test_complex_names)):
        # read cif file
        # data/complexes_16/xxx
        parser = MMCIFParser(QUIET=True)
        cplx_path = f"data/complexes_16/{complex_name}/pred.rank_0.cif"
        # if not exist, continue
        if not os.path.exists(cplx_path):
            print(f"File {cplx_path} not found, skipping...")
            continue
        structure = parser.get_structure(complex_name, cplx_path)
        coords_protein, features_protein, coords_ligand, features_ligand, b_factors_as_plddts = extractor.get_features(structure, "heuristic")
        center = np.mean(coords_ligand, axis=0)
        coords_cat = np.concatenate((coords_protein, coords_ligand), axis=0)
        features_cat = np.concatenate((features_protein, features_ligand), axis=0)
        coords_cat = coords_cat - center
        grid = extractor.grid(coords_cat, features_cat, n_amplification=9) # shape (10, 20, 20, 20, 28)
        grid = np.transpose(grid, (0, 4, 1, 2, 3)) # shape (10, 28, 20, 20, 20)
        test_grids.append(grid) # each grid is a tensor shape (10, 20, 20, 20, 28)
        test_labels.append(test_affinity[complex_name]) # each label is a float
        test_names.append(complex_name) # each name is a string
        test_plddt.append(np.min(b_factors_as_plddts)) # aggregate pLDDT by minimizing
        
    test_dataset = HFDataset.from_dict(
        {
            "grid": test_grids,
            "label": test_labels,
            "name": test_names,
            "pLDDT": test_plddt,
        },
        features=features,
    )

    # Save the dataset to disk
    test_dataset.save_to_disk(f"data/aug/pred_test_aug")

if __name__ == "__main__":
    create_real_dataset_aug()
    create_pred_dataset_aug()
    print("Done creating augmented datasets.")