import pathlib
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import regex as re
import matplotlib.pyplot as plt
import time
import random



class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source,target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
            )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
            )
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data)/(n_samples**2-n_samples)
        bandwidth/=kernel_mul**(kernel_num//2)
        bandwidth_list = [bandwidth*(kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance/bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0)-f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = torch.mean(kernels[:batch_size,:batch_size])
        YY = torch.mean(kernels[batch_size:,batch_size:])
        XY = torch.mean(kernels[:batch_size,batch_size:])
        YX = torch.mean(kernels[batch_size:,:batch_size])
        loss = torch.mean(XX+YY-XY-YX)
        return loss


'''
def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    #source covariance
    tmp_s = torch.ones((1,ns))#.to(DEVICE)
    print(f'tmp_s_shape',tmp_s.shape)
    a = tmp_s.t()@tmp_s
    print(f'a_shape', a.shape)
    cs = (source.t()@source-(tmp_s.t()@tmp_s)/ns)/(ns-1)

    #target covariance
    tmp_t = torch.ones((1,nt))#.to(DEVICE)
    ct = (target.t()@target-(tmp_t.t()@tmp_t)/nt)/(nt-1)

    #frobenius norm
    loss = (cs-ct).pow(2).sum().sqrt()
    loss = loss/(4*d*d)

    return loss
'''
def CORAL_loss(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)
 
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)
 
    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss




#A_distance
import numpy as np
from sklearn import svm
 
def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]
 
    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')
 
    C_list = np.logspace(-5, 4, 10)
 
    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))
 
    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))
 
    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)
 
        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)
 
        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))
 
        if test_risk > .5:
            test_risk = 1. - test_risk
 
        best_risk = min(best_risk, test_risk)
 
    return 2 * (1. - 2 * best_risk)




#Wasserstein loss
import math
import torch
import torch.linalg as linalg
 
def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''
 
    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")
 
    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)
 
    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact
 
    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()
 
    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar
 
    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar
 
    # put it together
    return mean_term



#LMMD
class LMMDLoss(MMDLoss):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, gamma = 1.0, max_iter=1000, **kwargs):
        super(LMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, **kwargs)
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.num_class = num_class

    def forward(self, source, target, source_label, target_logits):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")

        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            weight_ss = torch.from_numpy(weight_ss)
            weight_tt = torch.from_numpy(weight_tt)
            weight_st = torch.from_numpy(weight_st)

            kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma = self.fix_sigma)
            loss = torch.Tensor([0])
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]
            loss += torch.sum(weight_ss*SS+weight_tt*TT -2*weight_st*ST)
            return loss

    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label]   #one_hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 1000
        source_label_onehot = source_label_onehot/source_label_sum

        target_label = target_logits.cpu().data.max(1)[1].numpy()
        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits/target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class): #{B, C}
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)   #{B,1}
                t_tvec = target_logits[:,i].reshape(batch_size,-1)   #(B,1)

                ss = np.dot(s_tvec, s_tvec.T)   #(B,B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st+st
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss/length
            weight_tt = weight_tt/length
            weight_st = weight_st/length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


'''
a = torch.rand(4,32)
b = torch.rand(4,32)
a_label = torch.tensor([1,3,2,0],dtype=torch.long)
#a_label = F.one_hot(a_label, num_classes=4)
b_label = torch.rand(4,4) 
loss_function = LMMDLoss(num_class=4)
loss = loss_function(a,b, a_label, b_label)
print(loss)
'''

def DDC_loss(source_activation, target_activation):
	"""
	From the paper, the loss used is the maximum mean discrepancy (MMD)
	:param source: torch tensor: source data (Ds) with dimensions DxNs
	:param target: torch tensor: target data (Dt) with dimensons DxNt
	"""

	diff_domains = source_activation - target_activation
	loss = torch.mean(torch.mm(diff_domains, torch.transpose(diff_domains, 0, 1)))

	return loss

'''
def loss_margin(logit, label):
    loss = nn.CrossEntropyLoss()
    margin = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(logit.shape[0]):
        if label[i] == torch.tensor(0, dtype=torch.long):
            logit[i] = logit[i] + torch.tensor([-margin, margin, margin, margin],dtype=torch.float).to(DEVICE)
            #print(0)
        elif label[i] == torch.tensor(1, dtype=torch.long):
            logit[i] = logit[i] + torch.tensor([margin, -margin, margin, margin],dtype=torch.float).to(DEVICE)
            #print(1)
        elif label[i] == torch.tensor(2, dtype=torch.long):
            logit[i] = logit[i] + torch.tensor([margin, margin, -margin, margin], dtype=torch.float).to(DEVICE)
            #print(2)
        else:
            logit[i] = logit[i] + torch.tensor([margin, margin, margin, -margin], dtype=torch.float).to(DEVICE)
    return loss(logit, label)
'''


def loss_margin(logit, label):
    loss = nn.CrossEntropyLoss()
    margin = 0#0.6
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建与logit相同设备和类型的margin调整矩阵
    adjustment = torch.zeros_like(logit).to(DEVICE)
    adjustment.fill_(margin)  # 默认所有类别增加margin

    # 针对不同标签调整特定位置的margin
    adjustment[label == 0, 0] = -margin  # label0: 第0类减去margin
    adjustment[label == 1, 1] = -margin  # label1: 第1类减去margin
    adjustment[label == 2, 2] = -margin  #
    adjustment[label == 3, 3] = -margin
    #print(adjustment)

    # 应用调整 (原地操作保持原始行为)
    logit += adjustment

    return loss(logit, label)


class PairedCosineLoss(nn.Module):
    """
    成对余弦相似度损失函数
    L_pos = 4 - 2 * [cos(p1, g2) + cos(p2, g1)]

    输入:
        p1, p2: 预测向量组 (形状 [batch_size, feature_dim])
        g1, g2: 目标向量组 (形状 [batch_size, feature_dim])
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # 防止除零的小常数

    def forward(self, p1, p2, g1, g2):
        # 计算p1与g2的余弦相似度
        cos_sim_p1g2 = F.cosine_similarity(p1, g2, dim=1, eps=self.eps)

        # 计算p2与g1的余弦相似度
        cos_sim_p2g1 = F.cosine_similarity(p2, g1, dim=1, eps=self.eps)

        # 根据公式计算损失
        loss = 4 - 2 * (cos_sim_p1g2 + cos_sim_p2g1)

        # 返回批次平均损失
        return loss.mean()

