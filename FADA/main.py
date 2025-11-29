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
    'sourse_index':[2, 5, 8, 11, 1, 4, 7, 10],
    #'sourse_index':[0, 3, 6, 9,],
    'target_index':[0, 3, 6, 9],
    #'target_index':[ 2, 5, 8, 11],
    'cwt': 0,
    'sample_number': 1000,
    'sele_target' : 5, 
    'trans_h': 224, 'trans_w': 224, 'b_size': 32,
}


para_model = {
    'shuffer': 1,
    'chanels': 32,
    'lamud': 0.5,
    'num_classier': 1,
    'epochs': 80,
    'lr': 0.00001
 }


para_result = {
    'output_file': 'T11_3'
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
                  spe_list[s_i[4]][:n] + spe_list[s_i[5]][:n] + spe_list[s_i[6]][:n] + spe_list[s_i[7]][:n]  #源域文件列表
target_spe_list = spe_list[t_i[0]][:(n+t_n)] + spe_list[t_i[1]][:(n+t_n)] + spe_list[t_i[2]][:(n+t_n)] + spe_list[t_i[3]][:(n+t_n)]  #aokeng-2, hengwen-2, shuwen-2, wu-2

target_few_shot_1 = random.sample(spe_list[t_i[0]][:(n)], t_n)
target_few_shot_2 = random.sample(spe_list[t_i[1]][:(n)], t_n)
target_few_shot_3 = random.sample(spe_list[t_i[2]][:(n)], t_n)
target_few_shot_4 = random.sample(spe_list[t_i[3]][:(n)], t_n)
target_few_shot_spe_list = target_few_shot_1 + target_few_shot_2 + target_few_shot_3 + target_few_shot_4


print(f'f_s_spe_list:', len(target_few_shot_spe_list))
print(f'target_spe_list:', len(target_spe_list))
for i in range(len(target_few_shot_spe_list)):
    target_spe_list.remove(target_few_shot_spe_list[i])
    #print(len(target_spe_list))
#获得目标领域测试文件列表

def s_split_list(group):
    list_1 = spe_list[s_i[group]][:(n)] + spe_list[s_i[group+4]][:(n)]
    #print(f'list_1',len(list_1))
    list_2 = random.sample(list_1, int(n/2))
    #print(f'list_2',len(list_2))
    for i in range(len(list_2)):
        list_1.remove(list_2[i])
    #print(f'list_1',len(list_1))
    list_3 = random.sample(list_1, int(n/2))
    #print(f'list_3',len(list_3))
    return list_2, list_3

G1_01, G1_02 = s_split_list(0)
G1_11, G1_12 = s_split_list(1)
G1_21, G1_22 = s_split_list(2)
G1_31, G1_32 = s_split_list(3)
G1_1 = G1_01 + G1_11 + G1_21 + G1_31
G1_2 = G1_02 + G1_12 + G1_22 + G1_32



target_few_shot_1 = [item for _ in range(100) for item in target_few_shot_1]
target_few_shot_2 = [item for _ in range(100) for item in target_few_shot_2]
target_few_shot_3 = [item for _ in range(100) for item in target_few_shot_3]
target_few_shot_4 = [item for _ in range(100) for item in target_few_shot_4]
G2_12, _ = s_split_list(0)
G2_22, _ = s_split_list(1)
G2_32, _ = s_split_list(2)
G2_42, _ = s_split_list(3)
G2_1 = target_few_shot_1 + target_few_shot_2 + target_few_shot_3 + target_few_shot_4
G2_2 = G2_12 + G2_22 + G2_32 + G2_42



G3_11, G3_12 = s_split_list(0)
G3_21, G3_22 = s_split_list(1)
G3_31, G3_32 = s_split_list(2)
G3_41, G3_42 = s_split_list(3)
G3_1 = G3_11 + G3_21 + G3_31 + G3_41
G3_2 = G3_22[:165] + G3_32[:165] + G3_42[:170] +\
       G3_12[:170] + G3_32[165:165+165] + G3_42[170:170+165] +\
       G3_12[170:170+165] + G3_22[165:165+170] + G3_42[-165:] +\
       G3_12[-165:] + G3_22[-165:] + G3_32[-170:]



_, G4_12 = s_split_list(0)
_, G4_22 = s_split_list(1)
_, G4_32 = s_split_list(2)
_, G4_42 = s_split_list(3)
G4_1 = G2_1
G4_2 = G4_22[:165] + G4_32[:165] + G4_42[:170] +\
       G4_12[:170] + G4_32[165:165+165] + G4_42[170:170+165] +\
       G4_12[170:170+165] + G4_22[165:165+170] + G4_42[-165:] +\
       G4_12[-165:] + G4_22[-165:] + G4_32[-170:]

GZ_1 = G1_1 + G2_1 + G3_1 + G4_1
GZ_2 = G1_2 + G2_2 + G3_2 + G4_2
G1_label = torch.zeros(2000,dtype=torch.long)
G2_label = torch.ones(2000,dtype=torch.long)
G3_label = torch.ones(2000,dtype=torch.long)*2
G4_label = torch.ones(2000,dtype=torch.long)*3
GZ_label = torch.cat((G1_label,G2_label,G3_label,G4_label),dim=0)


G_ada1_label = torch.zeros(2000,dtype=torch.long)
G_ada3_label = torch.ones(2000,dtype=torch.long)*2
G_ada_1 = G2_1 + G4_1
G_ada_2 = G2_2 + G4_2
G_ada_label = torch.cat((G_ada1_label,G_ada3_label),dim=0)

target_fs_list = G2_1 + G2_1 + G2_1 + G2_1



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
from self_dataloader import get_dataloader,get_dataloader_G
g_ds = get_dataloader_G(GZ_1, GZ_2, GZ_label, transform, bs) #图1， 图2， 图1标签， 图2标签， 标签
g_ada_ds = get_dataloader_G(G_ada_1, G_ada_2, G_ada_label, transform, int(bs/2))
sourse_ds = get_dataloader(source_spe_list, transform, bs)
target_ds = get_dataloader(target_spe_list, transform, bs)
target_fs_ds = get_dataloader(target_fs_list, transform, bs)




#################################################################################################################################################################################
##############################################################################定义模型参数并训练模型###############################################################################
#################################################################################################################################################################################


#加载模型定义训练参数

from self_model import Model

model = Model(3)



# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()


# 开始训练
from self_train_gh import train
hist, y_true, y_pred, path = train(model, para_model['epochs'], g_ds, g_ada_ds, sourse_ds, target_ds, target_fs_ds, loss_fn, para_model['lr'], para_model['lamud'], para_result['output_file']) #, test_d2_for_CNN
from self_result import save_result_fig_and_Excel
from self_result import save_confusion_matrix
save_result_fig_and_Excel(path,hist)
conf_mat, f1_score = save_confusion_matrix(y_true, y_pred, path)

from self_train_d import train
hist, y_true, y_pred, path = train(model, para_model['epochs'], g_ds, g_ada_ds, sourse_ds, target_ds, target_fs_ds, loss_fn, para_model['lr'], para_model['lamud'], para_result['output_file']) #, test_d2_for_CNN
save_result_fig_and_Excel(path,hist)
conf_mat, f1_score = save_confusion_matrix(y_true, y_pred, path)



from self_train_adapt import train
hist, y_true, y_pred, path = train(model, para_model['epochs'], g_ds, g_ada_ds, sourse_ds, target_ds, target_fs_ds, loss_fn, para_model['lr'], para_model['lamud'], para_result['output_file']) #, test_d2_for_CNN
save_result_fig_and_Excel(path,hist)
conf_mat, f1_score = save_confusion_matrix(y_true, y_pred, path)
print(f'Macro F1 Score: {f1_score:.4f}')
print(f'end')



#if __name__ == '__main__':
#    main()