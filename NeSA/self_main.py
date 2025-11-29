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

# 设置数据集路径
para_dataload = {
    'cwt_path':[ 'aokeng-1/cwt_images', 'aokeng-2/cwt_images', 'aokeng-3/cwt_images',
                'hengwen-1/cwt_images', 'hengwen-2/cwt_images', 'hengwen-3/cwt_images',
                'shuwen-1/cwt_images', 'shuwen-2/cwt_images', 'shuwen-3/cwt_images',
                'wu-1/cwt_images', 'wu-2/cwt_images', 'wu-3/cwt_images' ],
    'sft_path':[ 'aokeng-1/SFT1_log_0_time_shift_0.001_0', 'aokeng-2/SFT1_log_0_time_shift_0.001_0', 'aokeng-3/SFT1_log_0_time_shift_0.001_0',
                'hengwen-1/SFT1_log_0_time_shift_0.001_0', 'hengwen-2/SFT1_log_0_time_shift_0.001_0', 'hengwen-3/SFT1_log_0_time_shift_0.001_0',
                'shuwen-1/SFT1_log_0_time_shift_0.001_0', 'shuwen-2/SFT1_log_0_time_shift_0.001_0', 'shuwen-3/SFT1_log_0_time_shift_0.001_0',
                'wu-1/SFT1_log_0_time_shift_0.001_0', 'wu-2/SFT1_log_0_time_shift_0.001_0', 'wu-3/SFT1_log_0_time_shift_0.001_0' ],
    'sourse_index':[0, 3, 6, 9, 1, 4, 7, 10],
    #'sourse_index':[0, 3, 6, 9,],
    'target_index':[2, 5, 8, 11],
    #'target_index':[ 2, 5, 8, 11],
    'cwt': 0,
    'sample_number': 1000,
    'sele_target' : 5, 
    'trans_h': 224, 'trans_w': 224, 'b_size': 16,
}


para_model = {
    'shuffer': 1,
    'chanels': 32,
    'Q_MMD': 0.5,
    'num_classier': 1,
    'epochs': 50,
    'lr': 0.0001
 }


para_result = {
    'output_file': 'T13_3'
 } 


########################################################################################################################################################################
#################################################################### 数据预处理和获取数据加载器##############################################################################
########################################################################################################################################################################

print(f'start')
# 创建结果文件夹
if not os.path.exists(para_result['output_file']):
    os.makedirs(para_result['output_file'])



# 获取文件路径并按数字排序返回频谱图列表
from self_dataloader import sort_by_number_in_filename
if int(para_dataload['cwt']):
    spe_path_list = [pathlib.Path(file_path) for file_path in para_dataload['cwt_path']]
    spe_list = [
    [str(path) for path in sorted(spe_path.glob('*.jpg'), key=sort_by_number_in_filename)]
    for spe_path in spe_path_list
]
else:
    spe_path_list = [pathlib.Path(file_path) for file_path in para_dataload['sft_path']]
    spe_list = [
    [str(path) for path in sorted(spe_path.glob('*.png'), key=sort_by_number_in_filename)]
    for spe_path in spe_path_list
]




# 获取源域、目标域和few-shot域的列表
n = para_dataload['sample_number']
s_i = para_dataload['sourse_index']
t_i = para_dataload['target_index']
t_n = para_dataload['sele_target']


source_spe_list = spe_list[s_i[0]][:n] + spe_list[s_i[1]][:n] + spe_list[s_i[2]][:n] + spe_list[s_i[3]][:n] + \
                  spe_list[s_i[4]][:n] + spe_list[s_i[5]][:n] + spe_list[s_i[6]][:n] + spe_list[s_i[7]][:n]  # aokeng-1, hengwen-1, shuwen-1, wu-1
target_spe_list = spe_list[t_i[0]][:(n+t_n)] + spe_list[t_i[1]][:(n+t_n)] + spe_list[t_i[2]][:(n+t_n)] + spe_list[t_i[3]][:(n+t_n)]  #aokeng-2, hengwen-2, shuwen-2, wu-2
target_few_shot_spe_list = random.sample(spe_list[t_i[0]][:(n)], t_n) + random.sample(spe_list[t_i[1]][:(n)], t_n) + \
                           random.sample(spe_list[t_i[2]][:(n)], t_n) + random.sample(spe_list[t_i[2]][:(n)], t_n)

for i in range(len(target_few_shot_spe_list)):
    target_spe_list.remove(target_few_shot_spe_list[i])
    #print(len(target_spe_list))
target_spe_list = target_spe_list + target_spe_list

#target_spe_list = target_spe_list + target_spe_list
print(f'target_spe_list:',len(target_spe_list))


#获取dataloader
img_height, img_width = para_dataload['trans_h'], para_dataload['trans_w']
if para_dataload['cwt']:
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width)),
    transforms.Normalize(mean=[0.2482, 0.7705, 0.7336],
                          std=[0.2897, 0.2375, 0.2974])
     ])
else:
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width)),
    transforms.Normalize(mean=[0.2700, 0.0616, 0.3794, 1],
                         std=[0.0129, 0.0766, 0.0526, 1])
     ])




bs = para_dataload['b_size']
from self_dataloader import get_dataloader
source_ds = get_dataloader(source_spe_list, transform, bs)
target_ds = get_dataloader(target_spe_list, transform, bs)
fs_ds = get_dataloader(target_few_shot_spe_list, transform, bs)




#################################################################################################################################################################################
##############################################################################定义模型参数并训练模型###############################################################################
#################################################################################################################################################################################



from self_model import ED_model
model = ED_model(3)


# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
loss_rec = nn.MSELoss(reduction='mean')
#ddc_loss = DDC_loss()

import torch
import torch.nn as nn
import math





# 开始训练
from self_train_pre import train_pre
from self_result import save_result_fig_and_Excel
from self_result import save_confusion_matrix

hist, y_true, y_pred, path = train_pre(model, para_model['epochs'], source_ds, target_ds, fs_ds, para_model['lr'], loss_fn, loss_rec, para_result['output_file']) #, test_d2_for_CNN
save_result_fig_and_Excel(path,hist)
conf_mat, f1_score = save_confusion_matrix(y_true, y_pred, path)
print(f'Macro F1 Score: {f1_score:.4f}')
print(f'end')


from self_train_f1 import train_f1
hist, y_true, y_pred, path = train_f1(model, para_model['epochs'], source_ds, target_ds, fs_ds, para_model['lr'], loss_fn, loss_rec, para_result['output_file']) #, test_d2_for_CNN
save_result_fig_and_Excel(path,hist)
conf_mat, f1_score = save_confusion_matrix(y_true, y_pred, path)
print(f'Macro F1 Score: {f1_score:.4f}')
print(f'end')


from self_train_f2 import train_f2
hist, y_true, y_pred, path = train_f2(model, para_model['epochs'], source_ds, target_ds, fs_ds, para_model['lr'], loss_fn, loss_rec, para_result['output_file']) #, test_d2_for_CNN
save_result_fig_and_Excel(path,hist)
conf_mat, f1_score = save_confusion_matrix(y_true, y_pred, path)
print(f'Macro F1 Score: {f1_score:.4f}')
print(f'end')