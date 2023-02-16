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
import math
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


from utils_gray import ImageToImage3D_Valid, Image2D, RandomFlip, RandomRotate, RandomContrast,ImageToImage3D_Pre
imgchant = 4

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

rf_val = RandomFlip(np.random.RandomState(), axis_prob=0,axis=0)
rr_val = RandomRotate(np.random.RandomState(), angle_spectrum=0)
rc_val = RandomContrast(np.random.RandomState(), alpha=(0.5, 1.5), mean=0.0, execution_probability=0)
val_dataset = ImageToImage3D_Pre(args.val_dataset,(imgdepth,imgsize,imgsize),None,None,None)
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
def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    #print(image.shape)
    c, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_z) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_xy) + 1
    #print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape[1:]).astype(np.float32)
    cnt = np.zeros(image.shape[1:]).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_z*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_xy * z, dd-patch_size[2])
                test_patch = image[:,xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch,axis=0).astype(np.float32)
                #print(test_patch.shape)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    return label_map, score_map

tbar_val = tqdm(valloader, ncols=100)
for batch_idx, (X_batch,*rest) in enumerate(tbar_val):
    # print(batch_idx)
    if isinstance(rest[0][0], str):
        image_filename = rest[0][0].replace('.npz','.nii.gz')
        image_filename = image_filename.replace('BraTS2021_','')	
        image_filename = image_filename.replace('_seg','')
    else:
        image_filename = '%s' % str(batch_idx + 1).zfill(3)

    #X_batch = Variable(X_batch.to(device='cuda'))
    # start = timeit.default_timer()
    with torch.no_grad():
      yHaT,_ = test_single_case(model,X_batch[0],60,40,(160,224,224),4)
      #y_out = model(X_batch)
      # stop = timeit.default_timer()
      # print('Time: ', stop - start) 
      #tmp = y_out.detach().cpu().numpy()

      #yHaT = np.argmax(tmp, axis=1)
      #anno_mat = np.full((155,240,240),0,dtype = np.int16)
      #anno_mat[:,8:232, 8:232] = yHaT[0,3:158,...]
      yHaT[yHaT == 3] = 4
    # print(fulldir+image_filename)
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    nii_file = sitk.GetImageFromArray(yHaT)
    sitk.WriteImage(nii_file,fulldir+image_filename) # nii_path 为保存路径





