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





def sort_by_number_in_filename(filename):
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    return match_numbers[0] if match_numbers else 0


# 自定义 Dataset 类
class Self_Dataset(Dataset):
    def __init__(self, file_list, labels_1, transform=None):
        self.file_list = file_list
        self.labels_1 = labels_1
        #self.labels_2 = labels_2
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform:
            img = self.transform(img)
        
        label_1 = self.labels_1[index]  # 获取标签
        #label_2 = self.labels_2[index]

        return img, label_1

    def __len__(self):
        return len(self.file_list)


class Self_Dataset_G(Dataset):
    def __init__(self, file_list1, file_list2, GZ_1_label, GZ_2_label, GZ_label, transform=None):
        self.file_list1 = file_list1
        self.file_list2 = file_list2
        self.GZ_1_label = GZ_1_label
        self.GZ_2_label = GZ_2_label
        self.GZ_label = GZ_label
        self.transform = transform
    def __getitem__(self, index):
        img_1 = Image.open(self.file_list1[index])
        img_2 = Image.open(self.file_list2[index])
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        GZ_1_label = self.GZ_1_label[index]  # 获取标签
        GZ_2_label = self.GZ_2_label[index]
        GZ_label = self.GZ_label[index]
        return img_1, img_2, GZ_1_label, GZ_2_label, GZ_label
    def __len__(self):
        return len(self.file_list1)



class Self_Dataset_domain(Dataset):
    def __init__(self, file_list, labels_1, labels_2, transform=None):
        self.file_list = file_list
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform:
            img = self.transform(img)
        
        label_1 = self.labels_1[index]  # 获取标签
        label_2 = self.labels_2[index]

        return img, label_1, label_2

    def __len__(self):
        return len(self.file_list)



def generate_labels(png_list):
    labels = []
    for path in png_list:
        # 根据文件夹路径来分配标签
        if 'aokeng' in str(path):  # aokeng -> 类别0
            labels.append(0)
        elif 'hengwen' in str(path):  # hengwen -> 类别1
            labels.append(1)
        elif 'shuwen' in str(path):  # shuwen -> 类别2
            labels.append(2)
        elif 'wu' in str(path):  # wu -> 类别3
            labels.append(3)
        else:
            raise ValueError(f"Unexpected path: {path}")  # 如果路径不符合预期
    
    # 将标签转换为 tensor 并使用 F.one_hot 转换为 one-hot 编码
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    #one_hot_labels = F.one_hot(labels_tensor, num_classes=4)  # 转换为 one-hot 编码
    #return one_hot_labels
    
    return labels_tensor #one_hot_labels


class Self_Dataset_G(Dataset):
    def __init__(self, file_list1, file_list2, GZ_1_label, GZ_2_label, GZ_label, transform=None):
        self.file_list1 = file_list1
        self.file_list2 = file_list2
        self.GZ_1_label = GZ_1_label
        self.GZ_2_label = GZ_2_label
        self.GZ_label = GZ_label
        self.transform = transform
    def __getitem__(self, index):
        img_1 = Image.open(self.file_list1[index])
        img_2 = Image.open(self.file_list2[index])
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        GZ_1_label = self.GZ_1_label[index]  # 获取标签
        GZ_2_label = self.GZ_2_label[index]
        GZ_label = self.GZ_label[index]
        return img_1, img_2, GZ_1_label, GZ_2_label, GZ_label
    def __len__(self):
        return len(self.file_list1)



def get_dataloader(spe_list, transform, batch_size):
    label = generate_labels(spe_list)
    #dataset = Self_Dataset(spe_list, label, transform)
    dataset = Self_Dataset(spe_list, label, transform)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #, num_workers=
    return dataset_loader


def get_dataloader_G(GZ_1, GZ_2, GZ_label, transform, batch_size):
    label_1 = generate_labels(GZ_1)
    label_2 = generate_labels(GZ_2)
    dataset = Self_Dataset_G(GZ_1, GZ_2, label_1, label_2, GZ_label, transform)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #, num_workers=
    return dataset_loader