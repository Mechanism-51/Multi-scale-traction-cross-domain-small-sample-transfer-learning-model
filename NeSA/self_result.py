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
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(x_arr, data[0], '-o', label='Train_loss')
    ax.plot(x_arr, data[1], '-s', label='Test_loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=7)
    ax.set_ylabel('Loss', size=7)
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(x_arr, data[2], '-o', label='Acc_train')
    ax.plot(x_arr, data[3], '-s', label='Acc_test')

    plt.savefig(os.path.join(path, 'Loss_and_accuracy.png'))
    data = np.transpose(np.vstack((np.array(data[0]),np.array(data[1]),np.array(data[2]),np.array(data[3]))))
    #print(f'hist_shape',hist.shape)
    df_hist = pd.DataFrame(data)
    df_hist.to_excel(os.path.join(path, 'Hist_loss_and_accuracy_cycle.xlsx'),index=False, header=['loss_Train_','loss_Test_','acc_Train', 'acc_Test'])


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
