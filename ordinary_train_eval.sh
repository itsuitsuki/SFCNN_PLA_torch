#!/bin/bash
# cd data_processing
# python create_ordinary_dataset.py
# cd ..
python train.py --n_epochs 200 # sfcnn paper: best epoch 112
python ordinary_eval.py