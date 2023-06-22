                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
# Code for MAT
import medpy.metric.binary as mmb
import torch
import lib
import argparse
import torch
import torchvision
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from thop import profile
import torch.nn.functional as F
import os
import gc
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit
import thop
from torch.cuda.amp import autocast, GradScaler
from lib.utils import adjust_learning_rate 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-warmup_epochs', '--warmup_epochs', default=10, type=int, metavar='N',
                    help='number of epochs for warmup (default: 20)')

parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument("--T", type=float)
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_freq', type=int,default = 4)

parser.add_argument('--modelname', default='gated3d', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--depth', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--kfold', default=5, type=int)

args = parser.parse_args()
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize
imgdepth = args.depth

from utils_gray import ImageToImage3D, Image2D, RandomFlip, RandomRotate, RandomContrast,ImageToImage3D_Convert,ImageToImage3D_SmallScale
imgchant = 4

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None
'''
tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)
'''

rf_train = RandomFlip(np.random.RandomState(), axis_prob=0.5,axis=0)
rf_val = RandomFlip(np.random.RandomState(), axis_prob=0.5,axis=0)
rr_train = RandomRotate(np.random.RandomState(), angle_spectrum=30)
rr_val = RandomRotate(np.random.RandomState(), angle_spectrum=30)
rc_train = RandomContrast(np.random.RandomState(), alpha=(0.5, 1.5), mean=0.0, execution_probability=0)
rc_val = RandomContrast(np.random.RandomState(), alpha=(0.5, 1.5), mean=0.0, execution_probability=0)
#predict_dataset = Image2D(args.val_dataset)
device = torch.device("cuda")



seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)
best_val=0
best_loss=99

def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert(predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)
def dice_coef_eachtumor(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert(predictions.shape == truth.shape)
    for i in range(num):
        for j in range(3):
            prediction = predictions[i,j,...]
            truth_ = truth[i,j,...]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores.append(1.0)
            else:
                scores.append((intersection + eps) / union)
    return scores

class DiceLoss(nn.Module):
    # calculeaza dice loss-ul
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability.shape == targets.shape)
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score
        
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets.float())
        
        return bce_loss + dice_loss


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
def dice_loss1(score, target):
    #target = target.float()
    smooth = 1e-5
    num_classes = 4
    weight = [1,1,1,1]
    loss_batch = 0
    batch = score.shape[0]
    onehot_label = np.zeros((target.shape[0], num_classes, target.shape[1], target.shape[2],target.shape[3]), dtype=np.float32)
    onehot_label = torch.from_numpy(onehot_label).cuda()
    score = F.softmax(score, dim=1).float()
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

criterion = LogNLLLoss()
a = 0
tem = args.T
for i in range(args.kfold):
    print('*' * 25, 'Num', i + 1, 'fold', '*' * 25)
    train_dataset = ImageToImage3D_SmallScale(Training = True ,shape = (imgdepth,imgsize,imgsize), kfold = args.kfold, i =i, dataset_path=args.train_dataset,  RandomFlip = rf_train,RandomRotate = rr_train,RandomContrast = rc_train )
    val_dataset = ImageToImage3D_SmallScale(Training = False ,shape = (imgdepth,imgsize,imgsize), kfold = args.kfold, i = i, dataset_path=args.train_dataset,  RandomFlip = rf_train,RandomRotate = rr_train,RandomContrast = rc_train)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if(args.kfold>1):
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    else:
        valloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    train_loss_sum, valid_loss_sum = 0, 0
    train_dice_sum , valid_dice_sum = 0, 0

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
        model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    model.to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.learning_rate,
                                 weight_decay=1e-5, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        if (epoch>=10):
            for param in model.parameters():
                param.requires_grad =True
        epoch_running_loss = 0
        epoch_dice = 0
        pre_data, pre_out = None, None
        tbar = tqdm(dataloader, ncols=100)
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(tbar):
          with torch.autograd.set_detect_anomaly(True):
            image , label = X_batch, y_batch
            X_batch = Variable(X_batch.to(device ='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # ===================forward=====================

            output = model(X_batch)
            #output = output.detach().cpu().numpy()


            if (pre_data != None) and (epoch>=50):
              pre_images, pre_label = pre_data
              if torch.cuda.is_available():
                  pre_images = pre_images.cuda()
                  pre_label = pre_label.cuda()
              out_pre = model(pre_images)
              loss1 = criterion(
                  torch.cat((out_pre, output), dim=0), torch.cat((pre_label, y_batch), dim=0)
                  )
              loss2 = dice_loss1(
                  torch.cat((out_pre, output), dim=0), torch.cat((pre_label, y_batch), dim=0)
                  )


              tmp2 = torch.cat((pre_label, y_batch), dim=0).detach().cpu().numpy()
              output_soft = F.softmax(torch.cat((out_pre, output), dim=0),dim = 1).detach().cpu().numpy()
              #loss2 = dice_loss1(output_soft, torch.cat((pre_label, y_batch), dim=0))

              dml_loss = (
                  F.kl_div(
                   F.log_softmax(out_pre / tem,dim = 1),
                   F.softmax(pre_out.detach() / tem,dim = 1),  # detach
                  reduction="mean",
                  )* tem* tem
                  )
              loss = 0.4*loss1 + 0.6*loss2+ 1* dml_loss
              #a += 1/(args.epochs-50)
              tem += 1/(args.epochs-50)	
            else:
                loss1 = criterion(output, y_batch)
                tmp2 =  y_batch.detach().cpu().numpy()
                output_soft = F.softmax(output,dim=1).detach().cpu().numpy()
                loss2 = dice_loss1(output, y_batch)

                loss = 0.4*loss1 + 0.6*loss2
            pre_data = (image , label)
            pre_out = output

            lr_next = adjust_learning_rate(args,optimizer,epoch,batch_idx,len(dataloader))
            tmp = output_soft
            yHaT = np.argmax(tmp, axis=1)
            #dice = mmb.dc(yHaT, tmp2)
            dice2 = dice_metric(yHaT, tmp2)
            epoch_dice += dice2

            # ===================backward====================


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_running_loss += loss.item()		
            tbar.set_description('epoch:{}, LR:{:.8f}, loss:{:.4f}, dice:{:.4f}  '.format(epoch, lr_next,loss,100*dice2))
        # ===================log========================
        print('epoch [{}/{}], Mean_loss:{:.4f}, dice:{:.6f}'
              .format(epoch, args.epochs, epoch_running_loss/(batch_idx+1), epoch_dice/(batch_idx+1)))
        train_loss_sum += epoch_running_loss/(batch_idx+1)
        train_dice_sum += epoch_dice/(batch_idx+1)




        if (epoch % args.val_freq) ==0:
            val_dice = 0
            val_loss = 0

            tbar_val = tqdm(valloader, ncols=100)
            for batch_idx, (X_batch, y_batch, *rest) in enumerate(tbar_val):
                # print(batch_idx)


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
                    loss =  0.4*loss1 + 0.6*loss2
                    dice = dice_metric(yHaT, yval)
                    val_dice += dice
                    val_loss += loss
                    '''
                    dice = dice_coef_metric(tmp, yval)
                    score = dice_coef_eachtumor(tmp, yval)
                    wt = score[0]
                    tc = score[1]
                    et = score[2]		
                    val_dice += dice
                    val_loss += loss
                    wt_count += wt
                    tc_count += tc
                    et_count += et
                    '''
                '''
                tmp[tmp>=0.5] = 1
                tmp[tmp<0.5] = 0
                tmp2[tmp2>0] = 1
                tmp2[tmp2<=0] = 0
                tmp2 = tmp2.astype(int)
                tmp = tmp.astype(int)
    
                # print(np.unique(tmp2))
                yHaT = tmp
                yval = tmp2
                '''
                epsilon = 1e-20

                del X_batch, y_batch,tmp,tmp2, y_out
                tbar.set_description('epoch:{}, loss:{:.4f}, dice:{:.4f}'.format(epoch, loss, 100 * dice))
                #tbar.set_description('epoch:{}, loss:{:.4f}, dice:{:.4f},WT:{:.4f},ET:{:.4f},TC:{:.4f},'.format(epoch, loss, 100 * dice, 100 * wt, 100 * tc, 100*et))
                #yHaT[yHaT==1] =255
                #yval[yval==1] =255
                fulldir = direc+"/{}/".format(epoch)
                # print(fulldir+image_filename)
                if not os.path.isdir(fulldir):

                    os.makedirs(fulldir)
            print("epoch [{}/{}],　Val_Loss:{:.4f}, Val_Dice:{:.4f}".format(epoch, args.epochs, val_loss/(batch_idx+1), val_dice/(batch_idx+1)))
            #print("epoch [{}/{}],　Val_Loss:{:.4f}, Val_Dice:{:.4f}, Mean_WT:{:.4f},Mean_ET:{:.4f},Mean_TC:{:.4f}".format(epoch, args.epochs, val_loss/(batch_idx+1), val_dice/(batch_idx+1),wt_count/(batch_idx+1), tc_count/(batch_idx+1),et_count/(batch_idx+1)))
                #cv2.imwrite(fulldir+image_filename, yHaT[0,:,:])
                # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
            valid_loss_sum += val_loss/(batch_idx+1)
            valid_dice_sum += val_dice/(batch_idx+1)
    fulldir = direc+"/{}/".format(epoch)
    torch.save(model.state_dict(), direc+"fold_"+str(i)+"_.pth")
    
print("Final results of" + str(args.kfold) + " folds :")
print('average train loss:{:.4f}, average train dice:{:.3f}%'.format(train_loss_sum / args.kfold,
                                                                             train_dice_sum / args.kfold))
print('average valid loss:{:.4f}, average valid dice:{:.3f}%'.format(valid_loss_sum / args.kfold,
                                                                             valid_dice_sum / args.kfold))
if(args.kfold == 1):
    fulldir = direc+"/{}/".format(epoch)
    torch.save(model.state_dict(), direc+"best_model.pth")
            
            


