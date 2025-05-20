#!/bin/bash
# cd data_processing
# python create_ordinary_dataset.py
# cd ..
CUDA_LAUNCH_BLOCKING=1 python train.py --n_epochs 1 --batch_size 256 # sfcnn paper: best epoch 112
CUDA_LAUNCH_BLOCKING=1 python ordinary_eval.py