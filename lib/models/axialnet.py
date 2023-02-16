import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import pdb
import matplotlib.pyplot as plt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
#from pyheatmap.heatmap import HeatMap
import random
#from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


'''
rpe_config = get_rpe_config(
    ratio=1.9,
    method="product",
    mode='ctx',
    shared_head=True,
    skip=1,
    rpe_on='k',
)
'''

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DepthWiseConv(nn.Module):
    def __init__ ( self,in_channel,out_channel,kernel_size = 3, stride = 1,padding = 1):
      #这一行千万不要忘记
        super(DepthWiseConv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv3d(in_channels=in_channel,
                  out_channels=in_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
 
        #逐点卷积
        self.point_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = input.clone()
        out = self.depth_conv(out)
        out = self.point_conv(out)
        return out


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
    def forward(self, x):
        # pdb.set_trace()
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False, length = False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.length = length
        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool3d(3, stride=(2,2,2),padding=1)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x): # N, C, L, H, W
        if self.length:
            x = x.permute(0, 3, 4, 1, 2) # N, H, W, C, L
        else:
          if self.width:
            x = x.permute(0, 2, 3, 1, 4)  # N, L, H, C, W
          else:
            x = x.permute(0, 2, 4, 1, 3)  # N, L, W, C, H
        N, L, W, C, H = x.shape
        
        x = x.contiguous().view(N * W * L, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * L * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W * L, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * L * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, L, W, self.out_planes, 2, H).sum(dim=-2)

        if self.length:
            output = output.permute(0, 3, 4, 1, 2)
        else:
          if self.width:
            output = output.permute(0, 3, 1, 2, 4)  # N, W, C, H
          else:
            output = output.permute(0, 3, 1, 4, 2)
        
        if self.stride > 1:
            output = self.pooling(output)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups )

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N*W,self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()


        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        # nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

#end of attn definition
class ConvAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False, length = False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.length = length
        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool3d(3, stride=(2,2,2),padding=1)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x): # N, C, L, H, W
        if self.length:
            x = x.permute(0, 3, 4, 1, 2) # N, H, W, C, L
        else:
          if self.width:
            x = x.permute(0, 2, 3, 1, 4)  # N, L, H, C, W
          else:
            x = x.permute(0, 2, 4, 1, 3)  # N, L, W, C, H
        N, L, W, C, H = x.shape
        
        x = x.contiguous().view(N * W * L, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * L * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W * L, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * L * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, L, W, self.out_planes, 2, H).sum(dim=-2)

        if self.length:
            output = output.permute(0, 3, 4, 1, 2)
        else:
          if self.width:
            output = output.permute(0, 3, 1, 2, 4)  # N, W, C, H
          else:
            output = output.permute(0, 3, 1, 4, 2)
        
        if self.stride > 1:
            output = self.pooling(output)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56, length=16):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(4,width)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, width=True)
        self.length_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size = length, stride=stride, length=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(4,planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.length_block(out)
        
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out

class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # print(kernel_size)
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size = 1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # pdb.set_trace()

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)

        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlock_conv(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56, length=16):
        super(AxialBlock_conv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = DepthWiseConv(planes, planes*self.expansion, kernel_size=5, stride=1, padding=2)
        self.bn1 = norm_layer(planes)
        self.small_block_1 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        self.small_block_2 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(1,3,1), stride=1, padding=(0,1,0))
        self.small_block_3 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(1,1,3), stride=1, padding=(0,0,1))
        self.normal_block_1 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(5,1,1), stride=1, padding=(2,0,0))
        self.normal_block_2 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(1,5,1), stride=1, padding=(0,2,0))
        self.normal_block_3 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(1,1,5), stride=1, padding=(0,0,2))
        self.big_block_1 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(7,1,1), stride=1, padding=(3,0,0))
        self.big_block_2 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(1,7,1), stride=1, padding=(0,3,0))
        self.big_block_3 = DepthWiseConv(planes*self.expansion, planes*self.expansion, kernel_size=(1,1,7), stride=1, padding=(0,0,3))        
        self.adjust_block = DepthWiseConv(planes, inplanes*self.expansion, kernel_size=3, stride=1, padding=1)
        self.conv_up = conv1x1(planes*self.expansion, planes) 
        self.conv_up1 = conv1x1(planes, planes)                  
        self.conv_up2 = conv1x1(inplanes*self.expansion, planes)
        self.bn2 = norm_layer(planes)
        self.gelu = nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        if self.downsample is not None:
          x = self.downsample(x)
        identity = x
        out = self.bn1(x)
        #out = self.conv_down(x)
        
        out = self.gelu(out)
        out_down = self.conv_down(out)
        

        small_out = self.small_block_1(out_down)
        small_out = self.small_block_2(small_out)
        small_out = self.small_block_3(small_out)
        normal_out = self.normal_block_1(out_down)
        normal_out = self.normal_block_2(normal_out)
        normal_out = self.normal_block_3(normal_out)
        big_out = self.big_block_1(out_down)
        big_out = self.big_block_2(big_out)
        big_out = self.big_block_3(big_out)
        out_down = small_out + big_out + out_down
        #out_down = small_out + normal_out + big_out + out_down

        out_down = self.conv_up(out_down)
        out += out_down
        #out = self.conv_up1(out)
        out = self.conv_up1(out)
        out += identity
        
        identity = out


        out = self.bn2(out)

        out = self.adjust_block(out)
        out = self.gelu(out)
        out = self.conv_up2(out)
        out += identity


        return out

#end of block definition
class ResAxialAttentionUNet3D(nn.Module):
    # model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    def __init__(self, block, layers, num_classes=4, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, img_depth = 128,imgchan=3):
        super(ResAxialAttentionUNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer

        self.inplanes = int(128 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(imgchan, self.inplanes, kernel_size=7, stride=(2,2,2), padding=3,
                               bias=False)
        self.conv2 = nn.Conv3d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(4,self.inplanes)
        self.bn2 = norm_layer(4,128)
        self.bn3 = norm_layer(4,self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], stride=1, kernel_size=(img_size // 4),length = img_depth//4)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 4),
                                       dilate=replace_stride_with_dilation[0],length = img_depth//4)
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 8),
                                       dilate=replace_stride_with_dilation[1],length = img_depth//8)
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=1, kernel_size=(img_size // 16),
                                       dilate=replace_stride_with_dilation[2],length = img_depth//16)
        #self.channel = conv1x1(int(64 * s),int(128 * s) , 1)
        # Decoder
        self.decoder1 = nn.Conv3d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=(2,2,2), padding=1)
        self.decoder2 = nn.Conv3d(int(1024  * s), int(1024 * s), kernel_size=3, stride=2, padding=1)
        self.decoder3 = nn.Conv3d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.decoder6 = nn.Conv3d(int(128 * s), int(64 * s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv3d(int(64 * s), num_classes, kernel_size=1, stride=1, padding=0)
        

        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False, length=16):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(4,planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size, length = length))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2
            length = length // 2
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size, length = length))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # AxialAttention Encoder
        # pdb.set_trace()

        
      x_slice = self.conv1(x)
      x_slice = self.bn1(x_slice)
        
      x_slice = self.relu(x_slice)
      x_slice = self.conv2(x_slice)
      x_slice = self.bn2(x_slice)
      x_slice = self.relu(x_slice)
      x_slice = self.conv3(x_slice)
      x_slice = self.bn3(x_slice)
      x_slice = self.relu(x_slice)
      x_bfpool = x_slice
      x_slice = self.pool(x_slice)

      x1 = self.layer1(x_slice)

      x2 = self.layer2(x1)

      x3 = self.layer3(x2)
      #x4 = self.layer4(x3)

      #x_slice = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2, 2, 2), mode='nearest'))

      #x_slice = torch.add(x_slice, x4)
      x_slice = F.relu(F.interpolate(self.decoder2(x3), scale_factor=(2, 2, 2), mode='trilinear'))
      x_slice = torch.add(x_slice, x3)
      x_slice = F.relu(F.interpolate(self.decoder3(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      x_slice = torch.add(x_slice, x2)
      x_slice = F.relu(F.interpolate(self.decoder4(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      x_slice = torch.add(x_slice, x1)
      x_slice = F.relu(F.interpolate(self.decoder5(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      #print(x_slice.shape)
      #print(x_bfpool.shape)
      #x_bfpool = self.channel(x_bfpool)
      #x_slice = torch.add(x_slice, x_bfpool)
      x_slice = F.relu(F.interpolate(self.decoder6(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      x_slice = self.adjust(F.relu(x_slice))
      # pdb.set_trace()
        
      
      return x_slice

    def forward(self, x):
        return self._forward_impl(x)

class ConvAttentionUNet3D(nn.Module):
    # model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    def __init__(self, block, layers, num_classes=4, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, img_depth = 128,imgchan=3):
        super(ConvAttentionUNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = int(128 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv3d(self.inplanes, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(16, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(16)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.GELU()
        #self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], stride=1, kernel_size=(img_size // 4))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 4),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 8),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size // 16),
                                       dilate=replace_stride_with_dilation[2])
        #self.channel = conv1x1(int(64 * s),int(128 * s) , 1)
        # Decoder
        self.decoder1 = nn.Conv3d(int(1024 * s), int(1024 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv3d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(int(128 * s), int(64 * s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv3d(int(64 * s), num_classes, kernel_size=1, stride=1, padding=0)
        

        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False, length=16):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size, length = length))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size, length = length))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # AxialAttention Encoder
        # pdb.set_trace()

        
      x_slice = self.conv1(x)
      x_slice = self.bn1(x_slice)
      x_slice = self.relu(x_slice)
      x_slice = self.conv2(x_slice)
      x_slice = self.bn2(x_slice)
      x_slice = self.relu(x_slice)
      x_slice = self.conv3(x_slice)
      x_slice = self.bn3(x_slice)
      x_slice = self.relu(x_slice)
      x_bfpool = x_slice
      #x_slice = self.pool(x_slice)

      x1 = self.layer1(x_slice)

      x2 = self.layer2(x1)

      x3 = self.layer3(x2)

      x4 = self.layer4(x3)
      
      x_slice = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2, 2, 2), mode='trilinear'))

      x_slice = torch.add(x_slice, x4)
      x_slice = F.relu(F.interpolate(self.decoder2(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      x_slice = torch.add(x_slice, x3)
      x_slice = F.relu(F.interpolate(self.decoder3(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      x_slice = torch.add(x_slice, x2)
      x_slice = F.relu(F.interpolate(self.decoder4(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      x_slice = torch.add(x_slice, x1)
      x_slice = F.relu(F.interpolate(self.decoder5(x_slice), scale_factor=(2, 2, 2), mode='trilinear'))
      #print(x_slice.shape)
      #print(x_bfpool.shape)
      #x_bfpool = self.channel(x_bfpool)
      #x_slice = torch.add(x_slice, x_bfpool)
      x_slice = self.adjust(x_slice)
      # pdb.set_trace()
        
      
      return x_slice

    def forward(self, x):
        return self._forward_impl(x)

class ConvAxialUNet3D(nn.Module):
    # model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    def __init__(self, block, block2,layers, num_classes=4, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, img_depth = 128,imgchan=3):
        super(ConvAxialUNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = int(16)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv3d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        #self.relu = nn.GELU()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(16), layers[0], stride=1, kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(32), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(1024 * s), layers[1], stride=2, kernel_size=(img_size // 4),
                                       dilate=replace_stride_with_dilation[0])
        self.layer4 = self._make_layer2(block2, int(1024 * s), layers[3], stride=2, kernel_size=(img_size // 8),
                                       dilate=replace_stride_with_dilation[2],length = img_depth//8)
        #self.channel = conv1x1(int(64 * s),int(128 * s) , 1)
        # Decoder
        self.decoder1 = nn.Conv3d(int(256), int(256), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv3d(int(256), int(128), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(int(128), int(32), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(int(32), int(16), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(int(16), int(8), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv3d(int(8), num_classes, kernel_size=1, stride=1, padding=0)
        

        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False, length=16):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size, length = length))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size, length = length))

        return nn.Sequential(*layers)
    def _make_layer2(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False, length=16):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size, length = length))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2
            length = length // 2
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size, length = length))

        return nn.Sequential(*layers)
    def _forward_impl(self, x):

        # AxialAttention Encoder
        # pdb.set_trace()

        
      x_slice = self.conv1(x)
      x_slice = self.bn1(x_slice)
      x_slice = self.relu(x_slice)
      x_slice = self.conv2(x_slice)
      x_slice = self.bn2(x_slice)
      x_slice = self.relu(x_slice)
      x_slice = self.conv3(x_slice)
      x_slice = self.bn3(x_slice)
      x_slice = self.relu(x_slice)


      x1 = self.layer1(x_slice)

      x2 = self.layer2(x1)

      x3 = self.layer3(x2)

      x4 = self.layer4(x3)

      x_slice = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2, 2, 2), mode='nearest'))

      x_slice = torch.add(x_slice, x4)
      x_slice = F.relu(F.interpolate(self.decoder2(x_slice), scale_factor=(2, 2, 2), mode='nearest'))

      x_slice = torch.add(x_slice, x3)
      x_slice = F.relu(F.interpolate(self.decoder3(x_slice), scale_factor=(2, 2, 2), mode='nearest'))
      x_slice = torch.add(x_slice, x2)
      x_slice = F.relu(F.interpolate(self.decoder4(x_slice), scale_factor=(2, 2, 2), mode='nearest'))
      #x_slice = torch.add(x_slice, x1)
      x_slice = F.relu(F.interpolate(self.decoder5(x_slice), scale_factor=(2, 2, 2), mode='nearest'))
      #print(x_slice.shape)
      #print(x_bfpool.shape)
      #x_bfpool = self.channel(x_bfpool)
      #x_slice = torch.add(x_slice, x_bfpool)
      x_slice = self.adjust(x_slice)
      # pdb.set_trace()
        
      
      return x_slice

    def forward(self, x):
        return self._forward_impl(x)

class ResAxialAttentionUNet3D_ultra(nn.Module):
    # model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    def __init__(self, block, layers, num_classes=4, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=128, imgchan=3):
        super(ResAxialAttentionUNet3D_ultra, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(imgchan, self.inplanes, kernel_size=7, stride=(2,2,2), padding=3,
                               bias=False)
        self.conv2 = nn.Conv3d(self.inplanes, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv3d(128, self.inplanes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], stride=1, kernel_size=(img_size // 4),length = img_depth//2)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0],length = img_depth//2)
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 4),
                                       dilate=replace_stride_with_dilation[1],length = img_depth//4)
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=1, kernel_size=(img_size // 8),
                                       dilate=replace_stride_with_dilation[2],length = img_depth//8)

        self.enconv1 = nn.Conv3d(int(64 * s), int(64 * s *2*2), kernel_size=3, stride=1, padding=1, bias=False)
        self.enconv2 = nn.Conv3d(int(128 * s*2), int(128 * s *2*2), kernel_size=3, stride=2, padding=1, bias=False)
        self.enconv3 = nn.Conv3d(int(256 * s*2), int(256 * s *2*2), kernel_size=3, stride=2, padding=1, bias=False)

        # Decoder
        self.decoder1 = nn.Conv3d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=(2,2,2), padding=1)
        self.decoder2 = nn.Conv3d(int(1024  * s), int(1024 * s), kernel_size=3, stride=2, padding=1)
        self.decoder3 = nn.Conv3d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.decoder6 = nn.Conv3d(int(128 * s), int(64 * s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv3d(int(64 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder_ad1 = conv1x1(int(128 * s),int(64 * s) , 1)
        self.decoder_ad2 = conv1x1(int(64 * s), int(128 * s), 1)
        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False, length=16):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size, length = length))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2
            length = length // 2
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size, length = length))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # AxialAttention Encoder
        # pdb.set_trace()

        
      x_slice = self.conv1(x)
      x_slice = self.bn1(x_slice)
        
      x_slice = self.relu(x_slice)
      x_slice = self.conv2(x_slice)
      x_slice = self.bn2(x_slice)
      x_slice = self.relu(x_slice)
      x_slice = self.conv3(x_slice)
      x_slice = self.bn3(x_slice)
      x_slice = self.relu(x_slice)
      x_bfpool = x_slice
      x_slice = self.pool(x_slice)
      
      x1 = self.layer1(x_slice)
      x_1 = self.enconv1(x_slice)

      x2 = self.layer2(x1)
      x_2 = self.enconv2(x_1)

      x3 = self.layer3(x2)
      x_3 = self.enconv3(x_2)
      #x4 = self.layer4(x3)
      x3 = torch.add(x3, x_3)
      #x_slice = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2, 2, 2), mode='nearest'))

      #x_slice = torch.add(x_slice, x4)
      x_slice = F.relu(F.interpolate(self.decoder2(x3), scale_factor=(2, 2, 2), mode='nearest'))
      x_slice = torch.add(x_slice, x3)
      x_slice = F.relu(F.interpolate(self.decoder3(x_slice), scale_factor=(2, 2, 2), mode='nearest'))
      x_slice = torch.add(x_slice, x2)
      x_slice = torch.add(x_slice, x_2)
      x_slice = F.relu(F.interpolate(self.decoder4(x_slice), scale_factor=(2, 2, 2), mode='nearest'))
      x_slice = torch.add(x_slice, x1)
      x_slice = torch.add(x_slice, x_1)
      x_slice = F.relu(F.interpolate(self.decoder5(x_slice), scale_factor=(2, 2, 2), mode='nearest'))
      #print(x_slice.shape)
      #print(x_bfpool.shape)
      x_slice = self.decoder_ad1(x_slice)
      x_slice = torch.add(x_slice, x_bfpool)
      x_slice = self.decoder_ad2(x_slice)
      x_slice = F.relu(F.interpolate(self.decoder6(x_slice), scale_factor=(2, 2, 2), mode='nearest'))
      x_slice = self.adjust(F.relu(x_slice))
      # pdb.set_trace()
        
      
      return x_slice

    def forward(self, x):
        return self._forward_impl(x)



def gated_3d(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet3D(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.25, **kwargs)
    return model
def gated_3d_ut(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet3D_ultra(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model
def AxialConv(pretrained=False, **kwargs):
    model = ConvAxialUNet3D(AxialBlock_conv, AxialBlock_dynamic,[1,2 , 4, 2], s= 0.25,  **kwargs)
    return model
def MedConv(pretrained=False, **kwargs):
    model = ConvAttentionUNet3D(AxialBlock_conv, [3, 3, 5, 2], s= 0.25,  **kwargs)
    return model

# EOF