import argparse
import lib
import torch
import torchvision
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import SimpleITK as sitk
import torch.nn.functional as F
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import numpy as np
import seaborn as sns
import pandas as pd
from hausdorff import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff
from monai.metrics import compute_hausdorff_distance

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 5)
parser.add_argument('--modelname', default='off', type=str,
                    help='name of the model to load')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--direc', default='./results', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loaddirec', default='load', type=str)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--depth', type=int, default=None)
parser.add_argument('--kfold', type=int, default=1)
args = parser.parse_args()

direc = args.direc
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize
loaddirec = args.loaddirec
imgdepth = args.depth
fulldir = "./visual/"

from utils_gray import ImageToImage3D, Image2D, RandomFlip, RandomRotate, RandomContrast,ImageToImage3D_Convert,ImageToImage3D_Valid,addjust_shape_mat
imgchant = 4

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

rf_val = RandomFlip(np.random.RandomState(), axis_prob=0,axis=0)
rr_val = RandomRotate(np.random.RandomState(), angle_spectrum=0)
rc_val = RandomContrast(np.random.RandomState(), alpha=(0.5, 1.5), mean=0.0, execution_probability=0)
val_dataset = ImageToImage3D_Valid(args.val_dataset,(imgdepth,imgsize,imgsize),None,None,None)
#predict_dataset = Image2D(args.val_dataset)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")

if modelname == "gated3d":
    model = lib.models.axialnet.gated_3d(img_size=imgsize, img_depth = imgdepth, imgchan=imgchant)
elif modelname == "gated3dut":
    model = lib.models.axialnet.gated_3d_ut(img_size=imgsize, img_depth = imgdepth,imgchan=imgchant)
elif modelname == "conv3d":
    model = lib.models.axialnet.MedConv(img_size=imgsize,img_depth = imgdepth, imgchan=imgchant)
elif modelname == "convtrans":
    model = lib.models.axialnet.AxialConv(img_size=imgsize,img_depth = imgdepth, imgchan=imgchant)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)
criterion = LogNLLLoss()
model.load_state_dict(torch.load(loaddirec))
def dice_metric(source, target):
    #target = target.float()
    smooth = 1e-5
    num_classes = 4
    loss = 0
    loss_batch = 0
    for i in range(num_classes):
      loss = 0
      for j in range(source.shape[0]):
        intersect = np.sum((source[j,:,:,:] == i) * (target[j,:,:,:] == i))
        z_sum = np.sum(source[j,:,:,:] == i )
        y_sum = np.sum(target[j,:,:,:] == i )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
      loss_batch += loss*1.0/source.shape[0]
    loss_final = loss_batch * 1.0 / num_classes
    return loss_final
def dice_score(probability , targets):
    smooth = 1e-5
    intersection = 2.0 * (probability * targets).sum()
    union = probability.sum() + targets.sum()
    dice_score = (intersection + smooth) / union

    return dice_score
def convert_wt(source, target):
    logits_WT = source.copy()
    mask_WT = target.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 3] = 1

    logits_WT[logits_WT == 1] = 1
    logits_WT[logits_WT == 2] = 1
    logits_WT[logits_WT == 3] = 1

    return logits_WT,mask_WT

def convert_tc(source, target):
    logits_TC = source.copy()
    mask_TC = target.copy()

    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 3] = 1

    logits_TC[logits_TC == 1] = 1
    logits_TC[logits_TC == 2] = 0
    logits_TC[logits_TC == 3] = 1

    return logits_TC,mask_TC

def convert_et(source, target):
    logits_ET = source.copy()
    mask_ET = target.copy()

    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 3] = 1

    logits_ET[logits_ET == 1] = 0
    logits_ET[logits_ET == 2] = 0
    logits_ET[logits_ET == 3] = 1

    return logits_ET,mask_ET

def dice_metric_full(source, target):
    #target = target.float()
    smooth = 1e-5
    num_classes = 4
    loss = 0
    loss_batch = 0

    logits_WT = source.copy()
    mask_WT = target.copy()

    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 3] = 1

    logits_WT[logits_WT == 1] = 1
    logits_WT[logits_WT == 2] = 1
    logits_WT[logits_WT == 3] = 1

    WT_score = dice_score(logits_WT,mask_WT)

    logits_TC = source.copy()
    mask_TC = target.copy()

    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 3] = 1

    logits_TC[logits_TC == 1] = 1
    logits_TC[logits_TC == 2] = 0
    logits_TC[logits_TC == 3] = 1

    TC_score = dice_score(logits_TC,mask_TC)

    logits_ET = source.copy()
    mask_ET = target.copy()

    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 3] = 1

    logits_ET[logits_ET == 1] = 0
    logits_ET[logits_ET == 2] = 0
    logits_ET[logits_ET == 3] = 1

    ET_score = dice_score(logits_ET,mask_ET)

    return WT_score,TC_score,ET_score
def dice_loss1(score, target):
    #target = target.float()
    smooth = 1e-5
    num_classes = 4
    weight = [1,1,1,1]
    loss_batch = 0
    batch = score.shape[0]
    onehot_label = np.zeros((target.shape[0], num_classes, target.shape[1], target.shape[2],target.shape[3]), dtype=np.float32)
    onehot_label = torch.from_numpy(onehot_label).cuda()

    for i in range(num_classes):
        onehot_label[:, i, :, :,:] = (target == i)

    for i in range(onehot_label.shape[1]):
      loss = 0
      for j in range(batch):
        intersect = torch.sum(score[j, i, ...] * onehot_label[j, i, ...])
        z_sum = torch.sum(score[j, i, ...] )
        y_sum = torch.sum(onehot_label[j, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
      loss_batch += loss*weight[i]/batch
    loss_final = 1 - loss_batch * 1.0 / onehot_label.shape[1]
    return loss_final
def acc_metric(source, target):
  acc = np.sum(source == target)
  acc = acc*1.0/(source.shape[0]*source.shape[1]*source.shape[2]*source.shape[3])
  return acc
def Hausdorff_dist(source, target):
    logits_WT = source.copy()
    mask_WT = target.copy()

    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 3] = 1

    logits_WT[logits_WT == 1] = 1
    logits_WT[logits_WT == 2] = 1
    logits_WT[logits_WT == 3] = 1

    HD_WT = compute_hausdorff_distance(np.expand_dims(logits_WT, axis=1),np.expand_dims(mask_WT, axis=1),include_background=True,percentile=95)


    logits_TC = source.copy()
    mask_TC = target.copy()

    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 3] = 1

    logits_TC[logits_TC == 1] = 1
    logits_TC[logits_TC == 2] = 0
    logits_TC[logits_TC == 3] = 1

    HD_TC = compute_hausdorff_distance(np.expand_dims(logits_TC, axis=1),np.expand_dims(mask_TC, axis=1),include_background=True,percentile=95)


    logits_ET = source.copy()
    mask_ET = target.copy()

    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 3] = 1

    logits_ET[logits_ET == 1] = 0
    logits_ET[logits_ET == 2] = 0
    logits_ET[logits_ET == 3] = 1

    HD_ET = compute_hausdorff_distance(np.expand_dims(logits_ET, axis=1),np.expand_dims(mask_ET, axis=1),include_background=True,percentile=95)


    print('HD95_WT:'+str(HD_WT[0][0]))
    print('HD95_TC:'+str(HD_TC[0][0]))
    print('HD95_ET:'+str(HD_ET[0][0]))
  
    return HD_WT[0][0],HD_TC[0][0],HD_ET[0][0]

val_dice = 0
val_dice_WT = 0
val_dice_TC = 0
val_dice_ET = 0
val_loss = 0
val_hd_WT = 0
val_hd_TC = 0
val_hd_ET = 0
tbar_val = tqdm(valloader, ncols=100)
for batch_idx, (X_batch, y_batch, *rest) in enumerate(tbar_val):
    # print(batch_idx)
    if isinstance(rest[0][0], str):
        image_filename = rest[0][0].replace('.npz','.nii.gz')
    else:
        image_filename = '%s' % str(batch_idx + 1).zfill(3)

    X_batch = Variable(X_batch.to(device='cuda'))
    y_batch = Variable(y_batch.to(device='cuda'))
    # start = timeit.default_timer()
    with torch.no_grad():
      y_out = model(X_batch)
      # stop = timeit.default_timer()
      # print('Time: ', stop - start) 
      tmp2 = y_batch.detach().cpu().numpy()
      tmp = y_out.detach().cpu().numpy()

      yHaT = np.argmax(tmp, axis=1)
      yval = tmp2
      loss1 = criterion(y_out, y_batch)
      loss2 = dice_loss1(y_out, y_batch)
      hd_wt,hd_tc,hd_et = Hausdorff_dist(yHaT,tmp2)
      loss = 0.4*loss1+0.6*loss2
      dice_wt,dice_tc,dice_et = dice_metric_full(yHaT, yval)
      dice = dice_metric(yHaT, yval)
      val_hd_WT += hd_wt
      val_hd_TC += hd_tc
      val_hd_ET += hd_et
      val_dice += dice
      val_dice_WT += dice_wt
      val_dice_TC += dice_tc
      val_dice_ET += dice_et
      val_loss += loss
      print(yHaT.shape)
      anno_mat = addjust_shape_mat(yHaT[0],(160,240,240))
      anno_mat = addjust_shape_mat(anno_mat,(155,240,240))
    # print(fulldir+image_filename)
    #if not os.path.isdir(fulldir):
        #os.makedirs(fulldir)
    #nii_file = sitk.GetImageFromArray(anno_mat)
    #sitk.WriteImage(nii_file,fulldir+image_filename) # nii_path 为保存路径
    tbar_val.set_description('loss:{:.4f}, WT_dice:{:.4f}, TC_dice:{:.4f},ET_dice:{:.4f},WT_HD:{:.4f},TC_HD:{:.4f},ET_HD:{:.4f}'.format( loss, 100 * dice_wt,100*dice_tc,100 * dice_et, hd_wt,hd_tc,hd_et))
print("Mean_Val_Dice:{:.4f},Mean_WT_Dice:{:.4f},Mean_TC_Dice:{:.4f},Mean_ET_Dice:{:.4f}, Mean_WT_HD:{:.4f}, Mean_TC_HD:{:.4f}, Mean_ET_HD:{:.4f}".format(val_dice / (batch_idx + 1) * 100,
                                        val_dice_WT / (batch_idx + 1) * 100,val_dice_TC / (batch_idx + 1) * 100,
                                        val_dice_ET / (batch_idx + 1) * 100,val_hd_WT/(batch_idx + 1),val_hd_TC/(batch_idx + 1),val_hd_ET/(batch_idx + 1) ))






