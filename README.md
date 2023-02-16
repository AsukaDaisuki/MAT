# MAT
A Lightweight Transformer Model for 3D Brain Tumor Segmentation
## Pre-processing
Please download BraTS2018 and BraTS2021 dataset.
## Training
*Run*
>python .train_ldb.py --train_dataset [your Training dataset]  --direc [your model checkpoints save to] --batch_size 1  --epoch 1 --val_freq 4 --modelname "gated3d" --T 1 --learning_rate 0.001 --imgsize 224 --depth 160 --kfold 5
>
## Testing
*Run*
>python ./test.py --val_dataset [your Testing dataset] --batch_size 1  --epoch 400 --save_freq 10 --modelname "gated3d" --loaddirec [your model checkpoints saved dir] --learning_rate 0.00005 --imgsize 224 --depth 160
>
## Predicting
*Run*
>python ./predict.py --val_dataset [your Predicting dataset] --batch_size 1  --epoch 400 --save_freq 10 --modelname "gated3d" --loaddirec [your model checkpoints saved dir] --learning_rate 0.00005 --imgsize 224 --depth 160
>
