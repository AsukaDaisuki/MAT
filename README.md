# MAT
A Lightweight Transformer Model for 3D Brain Tumor Segmentation
## Pre-processing
Please download BraTS2018 and BraTS2021 dataset.
## Training
Run
>python .train_lmb.py --train_dataset [your dataset]  --direc [your model checkpoints save to] --batch_size 1  --epoch 1 --val_freq 4 --modelname "gated3d" --T 1 --learning_rate 0.001 --imgsize 224 --depth 160 --kfold 5
>

