# SFCNN_PLA_torch

The implementation for Assessing the Reliability of AlphaFold3 Predictions for Protein-Ligand Affinity Prediction via [SFCNN](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04762-3). Also the repository for CS177 SP25 Course Project, ShanghaiTech. The original code for SFCNN is published [here](https://github.com/bioinfocqupt/Sfcnn) written in TensorFlow, while we adapt this into a version written by PyTorch.

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
