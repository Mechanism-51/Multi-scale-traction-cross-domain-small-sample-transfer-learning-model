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
def train(model, num_epochs, G_dataset, G_fs_dataset, source_ds, target_ds, target_fs_ds, loss_fn, lr,lamud, path_r):
    loss_hist_train = [0] * num_epochs
    loss_hist_test = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    accuracy_hist_test = [0] * num_epochs
    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        if epoch <30:
            model.g.requires_grad_(True)
            model.h.requires_grad_(True)
            model.d.requires_grad_(False)
            optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
            start_time = time.time()
            print(f'epoch<=30')
            #model.g.train()
            #model.h.train()
            for x_batch, y_batch in source_ds:
                x_batch = x_batch[:, :3, :, :].to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred, _, _ = model(x_batch,x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                is_correct = (
                        torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_train[epoch] += is_correct.sum().cpu()


        elif  epoch >= 30 and epoch <400:
            print(f'epoch30-400')
            model.g.requires_grad_(False)
            model.h.requires_grad_(False)
            model.d.requires_grad_(True)
            optimizer = torch.optim.Adam([{"params": model.d.parameters()}], lr=lr*10)
            start_time = time.time()
            for x_batch1, x_batch2, _, _, y_batch in G_dataset:
                x_batch1 = x_batch1[:, :3, :, :].to(DEVICE)
                x_batch2 = x_batch2[:, :3, :, :].to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                _, _, pred = model(x_batch1,x_batch2)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                is_correct = (
                        torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_train[epoch] += is_correct.sum().cpu()
            model.eval()
            with torch.no_grad():
                loss_source = 0
                accuracy_source = 0
                for x_batch, y_batch in source_ds:
                    x_batch = x_batch[:, :3, :, :].to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    pred, _, _ = model(x_batch, x_batch)
                    loss = loss_fn(pred, y_batch)
                    loss_source += loss.item() * y_batch.size(0)
                    is_correct = (
                            torch.argmax(pred, dim=1) == y_batch
                    ).float()
                    accuracy_source += is_correct.sum().cpu()
                loss_source = loss_source/len(source_ds.dataset)
                accuracy_source = accuracy_source/len(source_ds.dataset)
                print(f'loss_source:',loss_source)
                print(f'accuracy_source:',accuracy_source)

        else:
            print(f'epoch400-800')
            if epoch%2 == 0:
                model.g.requires_grad_(True)
                model.h.requires_grad_(True)
                model.d.requires_grad_(False)
                optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
                start_time = time.time()
                print(f'epoch_even')
                G_fs_iterator = iter(G_fs_dataset)
                target_fs_iterator = iter(target_fs_ds)
                for x_batch, y_batch in source_ds:
                    x_batch = x_batch[:, :3, :, :].to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    pred, _, _ = model(x_batch, x_batch)
                    loss1 = loss_fn(pred, y_batch)

                    x_batch_t, y_batch_t = next(target_fs_iterator)
                    x_batch_t = x_batch_t[:, :3, :, :].to(DEVICE)
                    y_batch_t = y_batch_t.to(DEVICE)
                    pred_t, _, _ = model(x_batch_t, x_batch_t)
                    loss2 = loss_fn(pred_t, y_batch_t) * 1/400

                    x_batch1, x_batch2, _, _, y_batch_G = next(G_fs_iterator)
                    x_batch1 = x_batch1[:, :3, :, :].to(DEVICE)
                    x_batch2 = x_batch2[:, :3, :, :].to(DEVICE)
                    y_batch_G = y_batch_G.to(DEVICE)
                    _, _, pred_G = model(x_batch1, x_batch2)
                    loss3 = loss_fn(pred_G, y_batch_G)
                    loss = loss1 + loss2 + loss3*lamud
                    loss.backward()
                    optimizer.step()
                    loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                    is_correct = (
                            torch.argmax(pred, dim=1) == y_batch
                    ).float()
                    accuracy_hist_train[epoch] += is_correct.sum().cpu()
            else:
                print(f'epoch_odd')
                model.g.requires_grad_(False)
                model.h.requires_grad_(False)
                model.d.requires_grad_(True)
                optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
                start_time = time.time()
                for x_batch1, x_batch2, _, _, y_batch in G_dataset:
                    x_batch1 = x_batch1[:, :3, :, :].to(DEVICE)
                    x_batch2 = x_batch2[:, :3, :, :].to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    _, _, pred = model(x_batch1, x_batch2)
                    loss = loss_fn(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                    is_correct = (
                            torch.argmax(pred, dim=1) == y_batch
                    ).float()
                    accuracy_hist_train[epoch] += is_correct.sum().cpu()
        #loss_hist_train[epoch] = loss_hist_train[epoch]/len(source_d1.dataset)/2
        loss_hist_train[epoch] /= len(source_ds.dataset)
        accuracy_hist_train[epoch] /= len(source_ds.dataset)

        if epoch>num_epochs-2:
            torch.save(model.state_dict(), os.path.join(path_r, 'model_{}.pt'.format(epoch+1)))
        # 验证阶段
        model.eval()
        all_y_true = []
        all_y_pred = []
        with torch.no_grad():
            for x_batch, y_batch in target_ds:
                x_batch = x_batch[:,:3,:,:].to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred, _, _ = model(x_batch, x_batch)
                loss = loss_fn(pred, y_batch)
                batch_preds = torch.argmax(pred, dim=1)
                if epoch > num_epochs-2:
                    all_y_true.extend(y_batch.cpu().numpy())
                    all_y_pred.extend(batch_preds.cpu().numpy())

                loss_hist_test[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                is_correct = is_correct.sum()
                accuracy_hist_test[epoch] += is_correct.cpu()
            loss_hist_test[epoch] /= len(target_ds.dataset)
            accuracy_hist_test[epoch] /= len(target_ds.dataset)

        # 输出每个 epoch 的结果
        end_time = time.time()
        cost_time = end_time - start_time
        print(f'Epoch {epoch + 1}/{num_epochs} - Time: {cost_time:.2f}s')
        print(f'Train Loss: {loss_hist_train[epoch]:.4f}, Train Accuracy: {accuracy_hist_train[epoch]:.4f}')
        print(f'Test Loss: {loss_hist_test[epoch]:.4f}, Test Accuracy: {accuracy_hist_test[epoch]:.4f}')


    return (loss_hist_train, loss_hist_test, accuracy_hist_train, accuracy_hist_test), all_y_true, all_y_pred