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

def train(model, num_epochs, source_d1, target_d1, target_fs_d1, lr_set,loss_fn, loss_fn1, critizen, ddc_loss,lamud, path_r):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    accuracy_hist_train_fs = [0] * num_epochs

    loss_hist_s = [0] * num_epochs
    loss_hist_fs = [0] * num_epochs
    loss_hist_t = [0] * num_epochs

    accuracy_hist_train_1 = [0] * num_epochs
    accuracy_hist_train_2 = [0] * num_epochs
    accuracy_hist_train_3 = [0] * num_epochs
    accuracy_hist_train_4 = [0] * num_epochs
    accuracy_hist_train_5 = [0] * num_epochs
    accuracy_hist_train_6 = [0] * num_epochs

    loss_train_d1 = [0] * num_epochs
    loss_train_d2 = [0] * num_epochs
    loss_train_d3 = [0] * num_epochs
    loss_train_d4 = [0] * num_epochs
    loss_train_d5 = [0] * num_epochs
    loss_train_d6 = [0] * num_epochs


    loss_hist_test = [0] * num_epochs
    accuracy_hist_test = [0] * num_epochs

    all_y_true = []
    all_y_pred = []

    model.to(DEVICE)



    for epoch in range(num_epochs):
        lr_1 = lr_set* math.pow(1, math.floor((1 + epoch) / 5))
        print(f'lr_1:',lr_1)
        optimizer = torch.optim.Adam([{"params": model.parameters()}],lr=lr_1)

        start_time = time.time()
        #model.train()
        #model.eval()
        if epoch <= 15:
            model.train()
        else:
            model.conv2.train()

        target_fs_iterator = iter(target_fs_d1)
        target_iterator = iter(target_d1)
        for x_batch, y_batch, s_label in source_d1:
            optimizer.zero_grad()
            x_batch = x_batch[:,:3,:,:].to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            output, logit1, logit2, logit3, logit4, logit5, logit6, ds1, ds2, ds3, ds4, ds5, ds6 = model(x_batch)
            loss_s = loss_fn(output[:, 0:4], y_batch)
            pred_s = F.one_hot(torch.argmax(output[:, 0:4], dim=1), num_classes=4)
            for i in range(1, int(output.shape[1] / 4)):
                loss_s += loss_fn(output[:, i * 4:(i + 1) * 4], y_batch)
                pred_s += F.one_hot(torch.argmax(output[:, i * 4:(i + 1) * 4], dim=1), num_classes=4)
            loss_s = loss_s + loss_fn(logit1, y_batch) + loss_fn(logit2, y_batch) + loss_fn(logit3, y_batch)+ loss_fn(logit4, y_batch) + loss_fn(logit5, y_batch) + loss_fn(logit6, y_batch)
            pred_s1 = F.one_hot(torch.argmax(logit1, dim=1), num_classes=4)
            pred_s2 = F.one_hot(torch.argmax(logit2, dim=1), num_classes=4)
            pred_s3 = F.one_hot(torch.argmax(logit3, dim=1), num_classes=4)
            pred_s4 = F.one_hot(torch.argmax(logit4, dim=1), num_classes=4)
            pred_s5 = F.one_hot(torch.argmax(logit5, dim=1), num_classes=4)
            pred_s6 = F.one_hot(torch.argmax(logit6, dim=1), num_classes=4)




            #上一行后留个位多层次分类器项
            x_batch_fs, y_batch_fs, _ = next(target_fs_iterator)
            x_batch_fs = x_batch_fs[:,:3,:,:][:x_batch.shape[0],:,:,:]
            y_batch_fs = y_batch_fs[:y_batch.shape[0]]
            x_batch_fs = x_batch_fs.to(DEVICE)
            #x_batch_fs = add_noise_image(x_batch_fs, 10)
            y_batch_fs = y_batch_fs.to(DEVICE)
            output_fs, logit1_fs, logit2_fs, logit3_fs, logit4_fs, logit5_fs, logit6_fs, _, _, _, _, _, _ = model(x_batch_fs)

            loss_fs = loss_fn(output_fs[:, 0:4], y_batch_fs)
            pred_fs = F.one_hot(torch.argmax(output_fs[:, 0:4], dim=1), num_classes=4)
            for i in range(1, int(output_fs.shape[1] / 4)):
                loss_fs += loss_fn(output_fs[:, i * 4:(i + 1) * 4], y_batch_fs)
                pred_fs += F.one_hot(torch.argmax(output_fs[:, i * 4:(i + 1) * 4], dim=1), num_classes=4)
            loss_fs = loss_fs+loss_fn(logit1_fs,y_batch_fs)+loss_fn(logit2_fs,y_batch_fs)+loss_fn(logit3_fs,y_batch_fs)+loss_fn(logit4_fs,y_batch_fs)+loss_fn(logit5_fs,y_batch_fs)+loss_fn(logit6_fs,y_batch_fs)



            #上一行后六项为多层次分类器项
            x_batch_t, _, t_label = next(target_iterator)
            x_batch_t = x_batch_t[:,:3,:,:].to(DEVICE)
            t_label = t_label.to(DEVICE)
            _, _, _, _, _, _, _, dt1, dt2, dt3, dt4, dt5, dt6 = model(x_batch_t)
            loss_t1 = critizen(ds1, dt1)
            loss_t2 = critizen(ds2, dt2)
            loss_t3 = critizen(ds3, dt3)
            loss_t4 = critizen(ds4, dt4)
            loss_t5 = critizen(ds5, dt5)
            loss_t6 = critizen(ds6, dt6)
            loss_t = loss_t1 + loss_t2 + loss_t3 + loss_t4 + loss_t5 + loss_t6

            if epoch <= 15:
                loss = loss_s + loss_fs + lamud * loss_t * (epoch + 1) / 50
            #loss = loss*math.pow(0.95, math.floor((1 + epoch) / 5))
            else:
                loss = loss_s + loss_fs + lamud * loss_t * (epoch + 1) / 50    # + lamud * loss_t * (epoch + 1) / 50
            loss.backward()
            optimizer.step()
            #optimizer.zero_grad()

            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            loss_hist_s[epoch] += loss_s.item() * y_batch.size(0)
            loss_hist_fs[epoch] += loss_fs.item() * y_batch.size(0)
            loss_hist_t[epoch] += loss_t.item() * y_batch.size(0)
            loss_train_d1[epoch] += loss_t1.item() * y_batch.size(0)
            loss_train_d2[epoch] += loss_t2.item() * y_batch.size(0)
            loss_train_d3[epoch] += loss_t3.item() * y_batch.size(0)
            loss_train_d4[epoch] += loss_t4.item() * y_batch.size(0)
            loss_train_d5[epoch] += loss_t5.item() * y_batch.size(0)
            loss_train_d6[epoch] += loss_t6.item() * y_batch.size(0)

            #print(loss)
            is_correct_s = (torch.argmax(pred_s, dim=1) == y_batch).float()
            is_correct = is_correct_s.sum()#+is_correct_fs.sum()
            accuracy_hist_train[epoch] +=is_correct.cpu()
            is_correct_fs = (torch.argmax(pred_fs, dim=1) == y_batch_fs).float()
            is_correct_fs = is_correct_fs.sum()#+is_correct_fs.sum()
            accuracy_hist_train_fs[epoch] +=is_correct_fs.cpu()


            is_correct_s1 = (torch.argmax(pred_s1, dim=1) == y_batch).float()
            is_correct1 = is_correct_s1.sum()#+is_correct_fs.sum()
            accuracy_hist_train_1[epoch] +=is_correct1.cpu()
            is_correct_s2 = (torch.argmax(pred_s2, dim=1) == y_batch).float()
            is_correct2 = is_correct_s2.sum()  # +is_correct_fs.sum()
            accuracy_hist_train_2[epoch] += is_correct2.cpu()
            is_correct_s3 = (torch.argmax(pred_s3, dim=1) == y_batch).float()
            is_correct3 = is_correct_s3.sum()  # +is_correct_fs.sum()
            accuracy_hist_train_3[epoch] += is_correct3.cpu()
            is_correct_s4 = (torch.argmax(pred_s4, dim=1) == y_batch).float()
            is_correct4 = is_correct_s4.sum()  # +is_correct_fs.sum()
            accuracy_hist_train_4[epoch] += is_correct4.cpu()
            is_correct_s5 = (torch.argmax(pred_s5, dim=1) == y_batch).float()
            is_correct5 = is_correct_s5.sum()  # +is_correct_fs.sum()
            accuracy_hist_train_5[epoch] += is_correct5.cpu()
            is_correct_s6 = (torch.argmax(pred_s6, dim=1) == y_batch).float()
            is_correct6 = is_correct_s6.sum()  # +is_correct_fs.sum()
            accuracy_hist_train_6[epoch] += is_correct6.cpu()







        #loss_hist_train[epoch] = loss_hist_train[epoch]/len(source_d1.dataset)/2 
        loss_hist_train[epoch] = loss_hist_train[epoch]/len(source_d1.dataset)
        loss_hist_s[epoch] = loss_hist_s[epoch]/len(source_d1.dataset)
        loss_hist_fs[epoch] = loss_hist_fs[epoch]/len(source_d1.dataset)
        loss_hist_t[epoch] = loss_hist_t[epoch]/len(source_d1.dataset)

        loss_train_d1[epoch] = loss_train_d1[epoch] / len(source_d1.dataset)
        loss_train_d2[epoch] = loss_train_d2[epoch] / len(source_d1.dataset)
        loss_train_d3[epoch] = loss_train_d3[epoch] / len(source_d1.dataset)
        loss_train_d4[epoch] = loss_train_d4[epoch] / len(source_d1.dataset)
        loss_train_d5[epoch] = loss_train_d5[epoch] / len(source_d1.dataset)
        loss_train_d6[epoch] = loss_train_d6[epoch] / len(source_d1.dataset)

        #accuracy_hist_train[epoch] = accuracy_hist_train[epoch]/len(source_d1.dataset)/2
        accuracy_hist_train[epoch] = accuracy_hist_train[epoch]/len(source_d1.dataset)
        accuracy_hist_train_fs[epoch] = accuracy_hist_train_fs[epoch]/len(source_d1.dataset)
        accuracy_hist_train_1[epoch] = accuracy_hist_train_1[epoch]/len(source_d1.dataset)
        accuracy_hist_train_2[epoch] = accuracy_hist_train_2[epoch] / len(source_d1.dataset)
        accuracy_hist_train_3[epoch] = accuracy_hist_train_3[epoch] / len(source_d1.dataset)
        accuracy_hist_train_4[epoch] = accuracy_hist_train_4[epoch] / len(source_d1.dataset)
        accuracy_hist_train_5[epoch] = accuracy_hist_train_5[epoch] / len(source_d1.dataset)
        accuracy_hist_train_6[epoch] = accuracy_hist_train_6[epoch] / len(source_d1.dataset)


        if epoch>num_epochs-2:
            torch.save(model.state_dict(), os.path.join(path_r, 'model_{}.pt'.format(epoch+1)))

        # 验证阶段
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, _ in target_d1:
                x_batch = x_batch[:,:3,:,:].to(DEVICE)
                #x_batch = add_noise_image(x_batch, 5)
                y_batch = y_batch.to(DEVICE)
                #print(f'x_batch',x_batch.shape)
                output, _, _, _, _, _, _, _, _, _, _, _, _ = model(x_batch)
                loss = loss_fn1(output[:,0:4],y_batch)
                pred = F.one_hot(torch.argmax(output[:,0:4], dim=1), num_classes=4)
                for i in range(1, int(output.shape[1]/4)):
                    #print(i)
                    loss += loss_fn1(output[:,i*4:(i+1)*4],y_batch)
                    pred += F.one_hot(torch.argmax(output[:,i*4:(i+1)*4], dim=1), num_classes=4)
                loss_hist_test[epoch] += loss.item()*y_batch.size(0)
                is_correct =  (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_test[epoch] += is_correct.sum().cpu()

                batch_preds = torch.argmax(pred, dim=1)
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


    return (loss_hist_train, loss_hist_test, loss_hist_s, loss_hist_fs, loss_hist_t,\
           loss_train_d1, loss_train_d2, loss_train_d3, loss_train_d4, loss_train_d5, loss_train_d6,\
           accuracy_hist_train, accuracy_hist_train_fs, accuracy_hist_test, \
           accuracy_hist_train_1, accuracy_hist_train_2, accuracy_hist_train_3, accuracy_hist_train_4, accuracy_hist_train_5,  accuracy_hist_train_6), \
           all_y_true, all_y_pred