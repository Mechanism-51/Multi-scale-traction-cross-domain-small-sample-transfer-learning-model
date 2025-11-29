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
import torchvision.models as models
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns






def save_result_fig_and_Excel(path,data):
    x_arr = np.arange(len(data[0])) + 1
    fig = plt.figure(figsize=(24,24))
    ax = fig.add_subplot(5, 1, 1)
    ax.plot(x_arr, data[0], '-o', label='Train_loss')
    ax.plot(x_arr, data[1], '-s', label='Test_loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=7)
    ax.set_ylabel('Loss', size=7)
    ax = fig.add_subplot(5, 1, 2)
    ax.plot(x_arr, data[2], '-o', label='Loss_hist_s')
    ax.plot(x_arr, data[3], '-s', label='Loss_hist_fs')
    ax.plot(x_arr, data[4], '-^', label='Loss_hist_t')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=7)
    ax.set_ylabel('Loss', size=7)

    ax = fig.add_subplot(5, 1, 3)
    ax.plot(x_arr, data[5], '-o', label='Loss_d1')
    ax.plot(x_arr, data[6], '-s', label='Loss_d2')
    ax.plot(x_arr, data[7], '-^', label='Loss_d3')
    ax.plot(x_arr, data[8], '-d', label='Loss_d4')
    ax.plot(x_arr, data[9], '-1', label='Loss_d5')
    ax.plot(x_arr, data[10], '-2', label='Loss_d6')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=7)
    ax.set_ylabel('Loss', size=7)

    ax = fig.add_subplot(5, 1, 4)
    ax.plot(x_arr, data[11], '-o', label='Acc_s')
    ax.plot(x_arr, data[12], '-s', label='Acc_fs')
    ax.plot(x_arr, data[13], '-^', label='Acc_test')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=7)
    ax.set_ylabel('Acc', size=7)

    ax = fig.add_subplot(5, 1, 5)
    ax.plot(x_arr, data[14], '-o', label='Acc_1')
    ax.plot(x_arr, data[15], '-s', label='Acc_2')
    ax.plot(x_arr, data[16], '-^', label='Acc_3')
    ax.plot(x_arr, data[17], '-d', label='Acc_4')
    ax.plot(x_arr, data[18], '-1', label='Acc_5')
    ax.plot(x_arr, data[19], '-2', label='Acc_6')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=7)
    ax.set_ylabel('Loss', size=7)

    plt.savefig(os.path.join(path, 'Loss_and_accuracy.png'))
    data = np.transpose(np.vstack((np.array(data[0]),np.array(data[1]),np.array(data[2]),np.array(data[3]),np.array(data[4]),np.array(data[5]),np.array(data[6]),\
                                   np.array(data[7]), np.array(data[8]), np.array(data[9]), np.array(data[10]),np.array(data[11]), np.array(data[12]), np.array(data[13]),\
                                   np.array(data[14]), np.array(data[15]), np.array(data[16]), np.array(data[17]), np.array(data[18]), np.array(data[19]))))
    #print(f'hist_shape',hist.shape)
    df_hist = pd.DataFrame(data)
    df_hist.to_excel(os.path.join(path, 'Hist_loss_and_accuracy_cycle.xlsx'),
                 index=False, header=['loss_Train_','loss_Test_','loss_hist_s', 'loss_hist_fs', 'loss_hist_t', 'loss_d1', 'loss_d2', 'loss_d3', 'loss_d4', 'loss_d5', 'loss_d6',\
                                      'acc_s', 'acc_fs', 'acc_test', 'acc_1', 'acc_2', 'acc_3', 'acc_4', 'acc_5', 'acc_6'])


def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(path, 'confusion_matrix.png'))
    plt.close()

    # 计算F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    with open(os.path.join(path, 'metrics.txt'), 'w') as f:
        f.write(f'Macro F1 Score: {f1:.4f}\n')

    return cm, f1
