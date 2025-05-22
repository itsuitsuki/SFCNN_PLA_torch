import os
import glob
import shutil
from datasets import Dataset as HFDataset
from datasets import Features, Array4D, Value, concatenate_datasets
from cif_parsing import FeatureExtractorCIF
import random
import openbabel as ob
from openbabel import pybel
import numpy as np
import torch
from tqdm import tqdm
from Bio.PDB import MMCIFParser
import warnings

# argparse
import argparse
parser = argparse.ArgumentParser(description="Create test dataset")
parser.add_argument(
    "--encoding",
    type=str,
    choices=["heuristic", "gold"],
    default="heuristic",
    help="Encoding type: only 'heuristic' or 'gold'"
)
args = parser.parse_args()

test_data_dir = "data/CASF-2016/coreset"
test_complex_names = glob.glob(os.path.join(test_data_dir, "*"))
test_complex_names = [os.path.basename(name) for name in test_complex_names]
# sort
test_complex_names.sort()

features = Features(
    {
        "grid": Array4D(dtype="int32", shape=(28, 20, 20, 20)),
        "label": Value(dtype="float32"), # (10,)
        "name": Value(dtype="string"), # (10,)
    }
)

extractor = FeatureExtractorCIF()
test_grids = []
test_labels = []
test_affinity = {}
test_names = []
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
    coords_protein, features_protein, coords_ligand, features_ligand = extractor.get_features(structure, args.encoding)
    center = np.mean(coords_ligand, axis=0)
    coords_cat = np.concatenate((coords_protein, coords_ligand), axis=0)
    features_cat = np.concatenate((features_protein, features_ligand), axis=0)
    coords_cat = coords_cat - center
    grid = extractor.grid(coords_cat, features_cat, n_amplification=0) # shape (1, 20, 20, 20, 28)
    grid = np.transpose(grid, (0, 4, 1, 2, 3)) # shape (1, 28, 20, 20, 20)
    test_grids.append(grid.squeeze(0)) # each grid is a tensor shape (20, 20, 20, 28)
    test_labels.append(test_affinity[complex_name]) # each label is a float
    test_names.append(complex_name) # each name is a string
    
test_dataset = HFDataset.from_dict(
    {
        "grid": test_grids,
        "label": test_labels,
        "name": test_names,
    },
    features=features,
)

# Save the dataset to disk
test_dataset.save_to_disk("data/heuristic_pred_test")