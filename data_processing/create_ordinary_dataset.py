import os
import glob
import shutil
from datasets import Dataset as HFDataset
from feature_utils import FeatureExtractor
import random
import openbabel as ob
from openbabel import pybel
import numpy as np
import torch
from tqdm import tqdm
import warnings
# warnings.filterwarnings("ignore")
# DATA AUGMENT -> 10 ROTATION FOR EACH TRAINING DATA COMPLEX
N_ROTATE = 10
N_SAMPLES_TRAIN = 4100 # 4100 * 10 = 41000 samples for training, others*10 for validation
SEED = 42
# Set random seed for reproducibility
random.seed(SEED)
# Set the random seed for numpy
np.random.seed(SEED)
# Set the random seed for PyTorch
torch.manual_seed(SEED)

train_data_dir = "../data/refined-set"
train_complex_names = glob.glob(os.path.join(train_data_dir, "*"))
train_complex_names = [os.path.basename(name) for name in train_complex_names]
# print("train_complex_names:", train_complex_names)

test_data_dir = "../data/CASF-2016/coreset"
test_complex_names = glob.glob(os.path.join(test_data_dir, "*"))
test_complex_names = [os.path.basename(name) for name in test_complex_names]
# sort
test_complex_names.sort()
# print("test_complex_names:", test_complex_names)

# deduplication
print("len(train_complex_names) before deduplication:", len(train_complex_names)-2)
# minus the intersection
train_complex_names = set(train_complex_names) - set(test_complex_names)
# delete 'index' and 'readme' from train_complex_names
train_complex_names = set(train_complex_names) - set(['index', 'readme'])
# to list
train_complex_names = list(train_complex_names)
# print("train_complex_names:", train_complex_names)
# print("len(train_complex_names):", len(train_complex_names))
print("len(test_complex_names):", len(test_complex_names))

# shuffle the train set and split into train/valid: 41000 for train, others for valid
train_complex_names.sort()
random.shuffle(train_complex_names)
valid_complex_names = train_complex_names[N_SAMPLES_TRAIN:]
train_complex_names = train_complex_names[:N_SAMPLES_TRAIN]

print("len(train_complex_names):", len(train_complex_names))
print("len(valid_complex_names):", len(valid_complex_names))

extractor = FeatureExtractor()

# train dir/index/INDEX_general_PL_data.2019
affinity_file = os.path.join(train_data_dir, "index", "INDEX_general_PL_data.2019")
train_affinity = {}
valid_affinity = {}
test_affinity = {}
# read affinity data
with open(affinity_file, "r") as f:
    for line in f.readlines():
        if line[0] != '#' and line.split()[0] in train_complex_names:
            # train_affinity[line.split()[0]] = float(line.split()[3])
            # 10 copies of the same affinity. name + str(i)
            for i in range(N_ROTATE):
                train_affinity[line.split()[0] + "_" + str(i)] = float(line.split()[3])
        elif line[0] != '#' and line.split()[0] in valid_complex_names:
            for i in range(N_ROTATE):
                valid_affinity[line.split()[0] + "_" + str(i)] = float(line.split()[3])
        elif line[0] != '#' and line.split()[0] in test_complex_names:
            test_affinity[line.split()[0]] = float(line.split()[3])

train_grids = []
train_labels = []
for complex_name in tqdm(train_complex_names, desc="Processing train complexes"):
    # read mol2 and pdb file
    mol2_file = os.path.join(train_data_dir, complex_name, complex_name + "_ligand.mol2")
    pdb_file = os.path.join(train_data_dir, complex_name, complex_name + "_protein.pdb")
    # print(mol2_file, pdb_file)
    ligand = pybel.readfile("mol2", mol2_file).__next__() # the very first one ligand molecule
    # print(ligand)
    protein = pybel.readfile("pdb", pdb_file).__next__() # the very first one protein molecule
    # print(protein)
    coords1, features1 = extractor.get_features(protein, 1) # protein, shape 
    coords2, features2 = extractor.get_features(ligand, 0) # ligand
    center = (np.max(coords2, axis=0) + np.min(coords2, axis=0)) / 2
    coords_cat = np.concatenate((coords1, coords2), axis=0)
    features_cat = np.concatenate((features1, features2), axis=0)
    assert coords_cat.shape[0] == features_cat.shape[0], "coords and features should have the same number of atoms"
    coords_cat = coords_cat - center
    grid = extractor.grid(coords_cat, features_cat) # shape (10, 20, 20, 20, 28)
    
    train_labels += [train_affinity[complex_name + "_" + str(i)] for i in range(N_ROTATE)]
    train_grids += [torch.tensor(grid[i], dtype=torch.float32) for i in range(N_ROTATE)] # each grid is a tensor shape (1, 20, 20, 20, 28)
train_dataset = HFDataset.from_dict({"grid": train_grids, "label": train_labels})
# save the train_affinity dict -> arrow
train_dataset.save_to_disk("../data/ordinary_dataset/train")

valid_grids = []
valid_labels = []
for complex_name in tqdm(valid_complex_names, desc="Processing valid complexes"):
    # read mol2 and pdb file
    mol2_file = os.path.join(train_data_dir, complex_name, complex_name + "_ligand.mol2")
    pdb_file = os.path.join(train_data_dir, complex_name, complex_name + "_protein.pdb")
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
    grid = extractor.grid(coords_cat, features_cat) # shape (10, 20, 20, 20)
    
    valid_labels += [valid_affinity[complex_name + "_" + str(i)] for i in range(N_ROTATE)]
    valid_grids += [torch.tensor(grid[i], dtype=torch.float32) for i in range(N_ROTATE)] # each grid is a tensor shape (20, 20, 20, 28)
valid_dataset = HFDataset.from_dict({"grid": valid_grids, "label": valid_labels})
# save the valid_affinity dict -> arrow
valid_dataset.save_to_disk("../data/ordinary_dataset/valid")


# stuff test set protein-ligand complexes into grids & save them
test_grids = []
test_labels = []
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
    grid = extractor.grid(coords_cat, features_cat, n_amplification=0) # shape (1, 20, 20, 20, 28)
    # save in the dict, torch
    # grid = torch.tensor(grid, dtype=torch.float32)

    test_grids.append(grid.squeeze(0)) # each grid is a tensor shape (20, 20, 20, 28)
    test_labels.append(test_affinity[complex_name])

# values of the test_affinity dict -> lists
test_dataset = HFDataset.from_dict({"grid": test_grids, "label": test_labels})
# save the test_affinity dict -> arrow
test_dataset.save_to_disk("../data/ordinary_dataset/test")
