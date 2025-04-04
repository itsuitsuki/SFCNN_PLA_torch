# SFCNN_PLA_torch

The implementation for Assessing the Reliability of AlphaFold3 Predictions for Protein-Ligand Affinity Prediction via [SFCNN](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04762-3). Also the repository for CS177 SP25 Course Project, ShanghaiTech. The original code for SFCNN is published [here](https://github.com/bioinfocqupt/Sfcnn) written in TensorFlow, while we adapt this into a version written by PyTorch.

# Preparation of Environment
## Create and activate a conda environment
Note that the environment is created with Python 3.7.5.

```sh
conda create -n sfcnn python=3.7.5
conda activate sfcnn
```

## Install dependencies
OpenBabel 2.4.1:
```sh
conda install -c conda-forge openbabel
```

PyTorch 1.3.1 (CUDA 12.4 version as the default in Apr 4 2025):
```sh
pip3 install torch torchvision torchaudio
```

Others: Please refer to `requirements.txt` for the full list of dependencies.
```sh
pip install -r requirements.txt
```

# Preparation of Data
## Download the dataset
The training set used in this project is PDBbind v2019 refined set. The test set is CASF-2016 scoring power benchmark.

PDBbind v2019:
```sh
wget -c https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz
```


CASF 2016:
```sh
wget -c https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/CASF-2016.tar.gz
```

## Unzip the dataset
```sh
mkdir data
tar -zxvf PDBbind_v2020_refined.tar.gz -C data
tar -zxvf CASF-2016.tar.gz -C data
rm PDBbind_v2020_refined.tar.gz
rm CASF-2016.tar.gz
```

## Create the direct-complex dataset
First, run the code to create the folders of direct complex structures in CASF-2016 core set.
```sh
cd data_processing
python create_complex_folders.py
```
Then, run AlphaFold 3 or CHAI-1 and generate the complex structures, which should be run in their own servers or separate local environments. You should place the generated complex structures in 
subfolders of `data/complexes_16`.

You should stay in the `data_processing` folder and run the following command:


## Create the protein-ligand complex dataset files (Arrow format)
Based on the last step, you are still in the `data_processing` folder. You can run the following command to create the ordinary dataset files (training & validation & testing) in Arrow format.
```sh
python create_ordinary_dataset.py
cd ..
```

Canonical dataset arrow files will be save in `data/ordinary_dataset` folder.

## File structure
```sh
data
├── CASF-2016
│   ├── coreset
│   │      ├── 1a30
│   │      │   ├── 1a30_ligand_opt.mol2
│   │      │   ├── 1a30_ligand.mol2
│   │      │   ├── 1a30_ligand.sdf
│   │      │   ├── 1a30_pocket.pdb
│   │      │   ├── 1a30_protein.mol2
│   │      │   └── 1a30_protein.pdb
│   │      └── ...
│   ├── ...
│   └── power_scoring
├── complexes_16
│   ├── 1a30
│   │   └── ...
│   └── ...
└── refined-set
    ├── 1a1e
    │   ├── 1a1e_ligand.mol2
    │   ├── 1a1e_ligand.sdf
    │   ├── 1a1e_pocket.pdb
    │   └── 1a1e_protein.pdb
    └── ...
```