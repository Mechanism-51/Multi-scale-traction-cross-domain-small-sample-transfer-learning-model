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
import math
from sklearn.metrics import confusion_matrix, f1_score





DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'GPU',torch.cuda.is_available())

# 定义训练过程

def train_adapt(model, num_epochs, source_d1, target_d1, lr_set, loss_fn, critizen, path_r, augm):
    num_epochs = 20
    print(f'adapt.......')
    path = os.path.join(path_r,'train_adapt')
    if not os.path.exists(path):
        os.makedirs(path)
    loss_hist_test = [0] * num_epochs
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    accuracy_hist_test = [0] * num_epochs
    all_y_true = []
    all_y_pred = []
    model.load_state_dict(torch.load(os.path.join(os.path.join(path_r, 'train_pre'), 'model_pre_20.pt')))
    model.to(DEVICE)
    
    for epoch in range(num_epochs):
        model.train()
        model.feature.requires_grad_(True)
        model.cla.requires_grad_(True)
        model.g.requires_grad_(True)
        model.p.requires_grad_(True)
        model.cla_f.requires_grad_(False)
        params = [{'params': model.feature.parameters(), 'lr': lr_set},
                  {'params': model.g.parameters(), 'lr': lr_set},
                  {'params': model.p.parameters(), 'lr': lr_set},
                  {'params': model.cla.parameters(), 'lr': lr_set}
                  ]
        optimizer = torch.optim.SGD(params,lr=lr_set,weight_decay=0.0001)
        start_time = time.time()
        target_iterator = iter(target_d1)
        for x_batch, y_batch in source_d1:
            optimizer.zero_grad()
            x_batch = x_batch[:, :3, :, :]
            x_batch1 = augm(x_batch)
            x_batch2 = augm(x_batch)
            x_batch1 = x_batch1.to(DEVICE)
            x_batch2 = x_batch2.to(DEVICE)
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            #print(x_batch1.shape)
            #print(x_batch2.shape)
            #print(x_batch.shape)

            y, p1, g2, _ = model(x_batch,x_batch1,x_batch2)
            y, p2, g1, _ = model(x_batch,x_batch2,x_batch1)
            loss1 = loss_fn(y,y_batch) +critizen(p1,p2,g1,g2)
            x_batch_t, y_batch_t = next(target_iterator)
            x_batch_t = x_batch_t[:, :3, :, :]
            X_batch_t1 = augm(x_batch_t)
            x_batch_t2 = augm(x_batch_t)
            x_batch_t1 = X_batch_t1.to(DEVICE)
            x_batch_t2 = x_batch_t2.to(DEVICE)
            x_batch_t = x_batch_t.to(DEVICE)
            _, p1t, g2t, _ = model(x_batch_t,x_batch_t1,x_batch_t2)
            _, p2t, g1t, _ = model(x_batch_t,x_batch_t2,x_batch_t1)
            loss2 = critizen(p1t,p2t,g1t,g2t)
            loss = 0.9*loss2*0.5 +0.1*loss1
            loss.backward()
            optimizer.step()

            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(y, dim=1) == y_batch).float()
            is_correct = is_correct.sum()#+is_correct_fs.sum()
            accuracy_hist_train[epoch] +=is_correct.cpu()       
        loss_hist_train[epoch] = loss_hist_train[epoch]/len(source_d1.dataset)
        accuracy_hist_train[epoch] = accuracy_hist_train[epoch]/len(source_d1.dataset)
        if epoch>num_epochs-2:
            torch.save(model.state_dict(), os.path.join(path, 'model_adapt_{}.pt'.format(epoch+1)))

        # 验证阶段
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in target_d1:
                x_batch = x_batch[:,:3,:,:].to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y, _, _, _ = model(x_batch,x_batch,x_batch)
                loss = loss_fn(y,y_batch)
                loss_hist_test[epoch] += loss.item()*y_batch.size(0)
                is_correct =  (
                    torch.argmax(y, dim=1) == y_batch
                ).float()
                accuracy_hist_test[epoch] += is_correct.sum().cpu()
                batch_preds = torch.argmax(y, dim=1)
                if epoch > num_epochs-2:
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(batch_preds.cpu().numpy())
            loss_hist_test[epoch] /= len(target_d1.dataset)
            accuracy_hist_test[epoch] /= len(target_d1.dataset)
        # 输出每个 epoch 的结果
        end_time = time.time()
        cost_time = end_time - start_time
        print(f'Epoch {epoch + 1}/{num_epochs} - Time: {cost_time:.2f}s')
        print(f'Train Loss: {loss_hist_train[epoch]:.4f}, Train Accuracy: {accuracy_hist_train[epoch]:.4f}')
        print(f'Test Loss 1: {loss_hist_test[epoch]:.4f}, Test Accuracy 1: {accuracy_hist_test[epoch]:.4f}')
    return (loss_hist_train, loss_hist_test, accuracy_hist_train, accuracy_hist_test), all_y_true, all_y_pred, path