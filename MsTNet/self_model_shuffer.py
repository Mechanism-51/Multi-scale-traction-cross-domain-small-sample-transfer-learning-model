import pathlib
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import regex as re
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from torchsummary import summary
import torchvision.models as models

#########################################################################################################################################################################################
#####################################################################################shufferNet 组件#####################################################################################
#########################################################################################################################################################################################
def channel_shuffle(x, groups):
    # 获得特征图的所以维度的数据
    batch_size, num_channels, height, width = x.shape
    # 对特征通道进行分组
    channels_per_group = num_channels // groups
    # reshape新增特征图的维度
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # 通道混洗(将输入张量的指定维度进行交换)
    x = torch.transpose(x, 1, 2).contiguous()
    # reshape降低特征图的维度
    x = x.view(batch_size, -1, height, width)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(ShuffleUnit, self).__init__()
        # 步长必须在1和2之间
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        # 输出通道必须能二被等分
        assert output_c % 2 == 0
        branch_features = output_c // 2

        # 当stride为1时，input_channel是branch_features的两倍
        # '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        # 捷径分支
        if self.stride == 2:
            # 进行下采样:3×3深度卷积+1×1卷积
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            # 不进行下采样:保持原状
            self.branch1 = nn.Sequential()

        # 主干分支
        self.branch2 = nn.Sequential(
            # 1×1卷积+3×3深度卷积+1×1卷积
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    # 深度卷积
    @staticmethod
    def depthwise_conv(input_c, output_c, kernel_s, stride, padding, bias= False):
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x):
        if self.stride == 1:
            # 通道切分
            x1, x2 = x.chunk(2, dim=1)
            # 主干分支和捷径分支拼接
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # 通道切分被移除
            # 主干分支和捷径分支拼接
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        # 通道混洗
        out = channel_shuffle(out, 2)
        return out
########################################################################################################################
##################################################   时频特征提取模块   ###################################################
########################################################################################################################
#输入的维度为（B,C,F,T）
#T维度减小到1/4，C维度扩大到2倍
class TD_X4(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.fd_f = nn.Sequential(nn.Conv2d(channels, channels*2, kernel_size=(1,7),stride=(1,4), padding=(0,2)),
				    nn.BatchNorm2d(channels*2),
				    nn.ReLU())

	def forward(self, X):
		Y = self.fd_f(X)
		return Y


class TD_X16(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.fd_f = nn.Sequential(nn.Conv2d(channels, channels*2, kernel_size=(1,7),stride=(1,4), padding=(0,2)),
				    nn.BatchNorm2d(channels*2),
				    nn.ReLU(),
					nn.Conv2d(channels*2, channels*4, kernel_size=(1, 7), stride=(1, 4),padding=(0, 2)),
					nn.BatchNorm2d(channels*4),
					nn.ReLU())

	def forward(self, X):
		Y = self.fd_f(X)
		return Y


#F维度减小到1/4，C维度扩大到2倍
class FD_X4(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.fd_f = nn.Sequential(nn.Conv2d(channels, channels*2, kernel_size=(7,1),stride=(4,1), padding=(2,0)),
				    nn.BatchNorm2d(channels*2),
				    nn.ReLU())

	def forward(self, X):
		Y = self.fd_f(X)
		return Y


class FD_X16(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.fd_f = nn.Sequential(nn.Conv2d(channels, channels*2, kernel_size=(7,1),stride=(4,1), padding=(2,0)),
				    nn.BatchNorm2d(channels*2),
				    nn.ReLU(),
					nn.Conv2d(channels*2, channels*4, kernel_size=(7, 1), stride=(4, 1), padding=(2, 0)),
					nn.BatchNorm2d(channels*4),
					nn.ReLU())

	def forward(self, X):
		Y = self.fd_f(X)
		return Y

class TFCM_T(nn.Module):
	def __init__(self, channels, m):
		super().__init__()
		self.pointwise_1 = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True))
		self.pointwise_2 = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True))
		self.depthwise_conv = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=(m,1), dilation=(m,1), groups=channels, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True))

	def forward(self, X):
		X = self.pointwise_1(X)
		X = self.pointwise_2(X)
		Y = self.depthwise_conv(X)
		return Y

class TFCM_F(nn.Module):
	def __init__(self, channels, m):
		super().__init__()
		self.pointwise_1 = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True))
		self.pointwise_2 = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True))
		self.depthwise_conv = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=(1,m), dilation=(1,m), groups=channels, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True))

	def forward(self, X):
		X = self.pointwise_1(X)
		X = self.pointwise_2(X)
		Y = self.depthwise_conv(X)
		return Y


def nearest_TU_X4(tensor_input):
	batch_size, channels, height, width = tensor_input.shape
	tensor_output = torch.zeros(batch_size, channels, height, width * 4, device=tensor_input.device)
	for i in range(4):
		tensor_output[:, :, :, i::4] = tensor_input
	return tensor_output


def nearest_TU_X16(tensor_input):
	batch_size, channels, height, width = tensor_input.shape
	tensor_output = torch.zeros(batch_size, channels, height, width * 16, device=tensor_input.device)
	for i in range(16):
		tensor_output[:, :, :, i::16] = tensor_input
	return tensor_output


def nearest_FU_X4(tensor_input):
	batch_size, channels, height, width = tensor_input.shape
	tensor_output = torch.zeros(batch_size, channels, height*4, width , device=tensor_input.device)
	for i in range(4):
		tensor_output[:, :, i::4, :] = tensor_input
	return tensor_output



def nearest_FU_X16(tensor_input):
	batch_size, channels, height, width = tensor_input.shape
	tensor_output = torch.zeros(batch_size, channels, height*16, width , device=tensor_input.device)
	for i in range(16):
		tensor_output[:, :, i::16, :] = tensor_input
	return tensor_output

class TU_X4(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.pointwise = nn.Sequential(
			nn.Conv2d(channels, channels//2, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels//2),
			nn.ReLU(inplace=True))

	def forward(self, X):
		X = self.pointwise(X)
		Y = nearest_TU_X4(X)
		return Y


class TU_X16(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.pointwise = nn.Sequential(
			nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels//4),
			nn.ReLU(inplace=True))

	def forward(self, X):
		X = self.pointwise(X)
		Y = nearest_TU_X16(X)
		return Y


class FU_X4(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.pointwise = nn.Sequential(
			nn.Conv2d(channels, channels//2, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels//2),
			nn.ReLU(inplace=True))

	def forward(self, X):
		X = self.pointwise(X)
		Y = nearest_FU_X4(X)
		return Y


class FU_X16(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.pointwise = nn.Sequential(
			nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
			nn.BatchNorm2d(channels//4),
			nn.ReLU(inplace=True))

	def forward(self, X):
		X = self.pointwise(X)
		Y = nearest_FU_X16(X)
		return Y


class Conv(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True))

	def forward(self, X):
		Y = self.conv(X)
		return Y


class MMF_T(nn.Module):
	def __init__(self, channels, m):
		super().__init__()
		self.conv = Conv(channels)
		self.tfcm_t_1 = ShuffleUnit(channels, channels, 1)
		self.tfcm_t_2 = ShuffleUnit(channels*2, channels*2, 1)
		self.tfcm_t_3 = ShuffleUnit(channels*4, channels*4, 1)
		self.td_X4_1 = TD_X4(channels)
		self.td_X4_2 = TD_X4(channels*2)
		self.td_X16 = TD_X16(channels)
		self.tu_X4_2 = TU_X4(channels*2)
		self.tu_X4_3 = TU_X4(channels*4)
		self.tu_X16 = TU_X16(channels * 4)
		self.relu = nn.ReLU(inplace=True)
		self.flatten = nn.Flatten()
		self.gap = nn.AdaptiveAvgPool2d((1, 1))
		self.gmp = nn.AdaptiveMaxPool2d((1, 1))
		self.class_X15 = nn.Linear(channels*2,4)
		self.class_X25 = nn.Linear(channels*2*2,4)
		self.class_X35 = nn.Linear(channels*4*2,4)
		self.drop = nn.Dropout(p=0.5)
	def forward(self, X):
		X11 = self.tfcm_t_1(self.conv(X))
		X21 = self.tfcm_t_2(self.td_X4_1(X))
		X12 = X11 + self.tu_X4_2(X21)
		X22 = X21 + self.td_X4_1(X11)
		X13 = self.tfcm_t_1(X12)
		X23 = self.tfcm_t_2(X22)
		X33 = self.tfcm_t_3(self.td_X4_2(X22))
		X14 = X13 + self.tu_X4_2(X23) + self.tu_X16(X33)
		X24 = self.td_X4_1(X13) + X23 + self.tu_X4_3(X33)
		X34 = self.td_X16(X13) + self.td_X4_2(X23) + X33
		X15 = self.tfcm_t_1(X14)
		X25 = self.tfcm_t_2(X24)
		X35 = self.tfcm_t_3(X34)
		Y = X15 + self.tu_X4_2(X25) +self.tu_X16(X35)
		X15_1 = self.gap(X15)
		X15_2 = self.gmp(X15)
		X15 = self.relu(torch.cat((self.flatten(X15_1), self.flatten(X15_2)), dim=1))
		X15 = self.drop(X15)
		Y15 = self.class_X15(X15)
		X25_1 = self.gap(X25)
		X25_2 = self.gmp(X25)
		X25 = self.relu(torch.cat((self.flatten(X25_1), self.flatten(X25_2)), dim=1))
		X25 = self.drop(X25)
		Y25 = self.class_X25(X25)
		X35_1 = self.gap(X35)
		X35_2 = self.gmp(X35)
		X35 = self.relu(torch.cat((self.flatten(X35_1), self.flatten(X35_2)), dim=1))
		X35 = self.drop(X35)
		Y35 = self.class_X35(X35)
		return Y, Y15, Y25, Y35, X15, X25, X35 



class MMF_F(nn.Module):
	def __init__(self, channels, m):
		super().__init__()
		self.conv = Conv(channels)
		self.tfcm_f_1 = ShuffleUnit(channels, channels, 1)
		self.tfcm_f_2 = ShuffleUnit(channels*2, channels*2, 1)
		self.tfcm_f_3 = ShuffleUnit(channels*4, channels*4, 1)                                                   
		self.fd_X4_1 = FD_X4(channels)
		self.fd_X4_2 = FD_X4(channels*2)
		self.fd_X16 = FD_X16(channels)
		self.fu_X4_2 = FU_X4(channels*2)
		self.fu_X4_3 = FU_X4(channels*4)
		self.fu_X16 = FU_X16(channels * 4)
		self.relu = nn.ReLU(inplace=True)
		self.flatten = nn.Flatten()
		self.gap = nn.AdaptiveAvgPool2d((1, 1))
		self.gmp = nn.AdaptiveMaxPool2d((1, 1))
		self.class_X15 = nn.Linear(channels*2,4)
		self.class_X25 = nn.Linear(channels*2*2,4)
		self.class_X35 = nn.Linear(channels*4*2,4)
		self.drop1 = nn.Dropout(p=0.5)
		self.drop2 = nn.Dropout(p=0.5)
		self.drop3 = nn.Dropout(p=0.5)
	def forward(self, X):
		X11 = self.tfcm_f_1(self.conv(X))
		X21 = self.tfcm_f_2(self.fd_X4_1(X))
		X12 = X11 + self.fu_X4_2(X21)
		X22 = X21 + self.fd_X4_1(X11)
		X13 = self.tfcm_f_1(X12)
		X23 = self.tfcm_f_2(X22)
		X33 = self.tfcm_f_3(self.fd_X4_2(X22))
		X14 = X13 + self.fu_X4_2(X23) + self.fu_X16(X33)
		X24 = self.fd_X4_1(X13) + X23 + self.fu_X4_3(X33)
		X34 = self.fd_X16(X13) + self.fd_X4_2(X23) + X33
		X15 = self.tfcm_f_1(X14)
		X25 = self.tfcm_f_2(X24)
		X35 = self.tfcm_f_3(X34)
		Y = X15 + self.fu_X4_2(X25) +self.fu_X16(X35)
		X15_1 = self.gap(X15)
		X15_2 = self.gmp(X15)
		X15 = self.relu(torch.cat((self.flatten(X15_1), self.flatten(X15_2)), dim=1))
		X15 = self.drop1(X15)
		Y15 = self.class_X15(X15)
		X25_1 = self.gap(X25)
		X25_2 = self.gmp(X25)
		X25 = self.relu(torch.cat((self.flatten(X25_1), self.flatten(X25_2)), dim=1))
		X25 = self.drop2(X25)
		Y25 = self.class_X25(X25)
		X35_1 = self.gap(X35)
		X35_2 = self.gmp(X35)
		X35 = self.relu(torch.cat((self.flatten(X35_1), self.flatten(X35_2)), dim=1))
		X35 =self.drop3(X35)
		Y35 = self.class_X35(X35)
		return Y, Y15, Y25, Y35, X15, X25, X35 



#model = MMF_T(256,1)
#input = torch.rand(1,256,16,16)
#output = model(input)
#print(output.shape)



#########################################################################################################################################################################################
#########################################################################################################################################################################################


class Model(nn.Module):
	def __init__(self, inputchannels, channels, m, num_classier):
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True)
		)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.mmf_t = MMF_T(channels, 1)
		self.mmf_f = MMF_F(channels, 1)
		#self.cmf = CMF(16)
		#self.cbam = CBAMLayer(32)
		#self.bn = nn.BatchNorm2d(512)
		self.gap = nn.AdaptiveAvgPool2d((1, 1))
		self.gmp = nn.AdaptiveMaxPool2d((1, 1))
		self.flatten = nn.Flatten()
		self.relu = nn.ReLU(inplace=True)
		self.drop = nn.Dropout(p=0.1)
		self.classifer = nn.Linear(channels*4,4*num_classier)
		

	def forward(self, X):
		X = self.conv1(X)
		X = self.maxpool(X)
		Xt,T15,T25,T35,Xt_15,Xt_25,Xt_35 = self.mmf_t(X)      #最后一层分类器的输入特征，多尺度层的分类结果*3，多尺度分类器的输入特征*3
		Xf,F15,F25,F35,Xf_15,Xf_25,Xf_35 = self.mmf_f(X)
		X1 = self.gap(Xt)#.squeeze()
		X2 = self.gmp(Xt)#.squeeze()
		X3 = self.gap(Xf)#.squeeze()
		X4 = self.gmp(Xf)#.squeeze()
		X = torch.cat((self.flatten(X1), self.flatten(X2), self.flatten(X3), self.flatten(X4)), dim=1)
		X = self.drop(X)
		Y = self.classifer(X)
		return Y, T15, T25, T35, F15, F25, F35, X, Xt_15, Xt_25, Xt_35, Xf_15, Xf_25, Xf_35  #最后一层集成分类器的结果，多尺度分类器的结果*6，最后一层分类器的输入特征，多尺度分类器的输入特征*6