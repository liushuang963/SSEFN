import os
import sys
import time
import math
import torch
import random
import numpy as np
from sklearn import metrics

import utils.evaluation as evaluation
import utils.data_load_operate as data_load_operate
import model.SSEFN as SSEFN
import argparse
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from utils.data_load_operate import HSI_create_pathes


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_flag',type=int,default=0)
    parser.add_argument('--use_att', type=bool, default=True)
    parser.add_argument('--dropout_rate',type=float,default=0.2)
    parser.add_argument('--save_folder_name', type=str, default="EXP")
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--batch_size',type=int,default=24)
    parser.add_argument('--EPOCHS',type=int,default=100)
    parser.add_argument('--train_num',type=int,default=5)
    parser.add_argument('--val_num',type=int,default=5)
    parser.add_argument('--fusion_mode',type=str,default='decision')
    return parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_parser()
model_name = ['SSEFN']

model_flag = args.model_flag

model_3D_spa_flag = 0
model_type_flag = 1


data_set_name = 'IP'


ratio = 10.0
patch_size = 27


args.patch_size = patch_size
use_att = args.use_att
patch_length = int((patch_size - 1) / 2)

data_set_path = "./data"
work_dir = "./work_dir/" + args.save_folder_name


if __name__ == '__main__':

    torch.cuda.empty_cache()

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    if model_flag == 1:
        data = data_load_operate.applyPCA(data, 30)

    height, width, channels = data.shape

    data = data_load_operate.standardization(data)

    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    class_count = max(np.unique(gt))

    flag_list = [1, 0]  #
    ratio_list = [0.1, 0.01]

    train_num = args.train_num
    val_num = args.val_num
    num_list = [train_num, val_num]

    if model_flag in [2,3,4]:
        batch_size = 128
    else:
        batch_size = args.batch_size
    max_epoch = args.EPOCHS
    if model_flag in [2,5]:
        max_epoch = 2 * max_epoch
    learning_rate = args.learning_rate
    loss = torch.nn.CrossEntropyLoss()

    data_padded = data_load_operate.data_pad_zero(data, patch_length)
    height_patched, width_patched, channels = data_padded.shape


    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    save_experiment_dir = os.path.join(work_dir, 'SSEFN')

    experiment_name = 'run'

    if not os.path.exists(save_experiment_dir):
        os.makedirs(save_experiment_dir)

    experiment_name = 'run'
    save_experiment_folder = os.path.join(save_experiment_dir,experiment_name)
    if not os.path.exists(save_experiment_folder):
        os.mkdir(save_experiment_folder)
    save_weight_path = os.path.join(save_experiment_folder,'best.pth')
    save_result_path = os.path.join(save_experiment_folder,'result.txt')
    save_predict_path = os.path.join(save_experiment_folder,'predict.png')
    save_gt_path = os.path.join(save_experiment_folder,'gt.png')

    train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                                   num_list,
                                                                                                   gt_reshape,
                                                                                                   class_count,
                                                                                                   flag_list[0])

    index = (train_data_index, val_data_index, test_data_index)
    train_iter, test_iter, val_iter = data_load_operate.generate_iter_1 \
        (data_padded, height, width, gt_reshape, index, patch_length, batch_size, model_type_flag,
         model_3D_spa_flag)


    net = SSEFN.SSEFN(in_channels=channels,num_classes=class_count,use_att=use_att)

    net.to(device)


    train_loss_list = [100]
    train_acc_list = [0]
    val_loss_list = [100]
    val_acc_list = [0]
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    best_loss = 99999

    tic1 = time.perf_counter()

    for epoch in range(max_epoch):

        train_acc_sum, trained_samples_counter = 0.0, 0
        batch_counter, train_loss_sum = 0, 0
        time_epoch = time.time()
        net.train()
        if model_type_flag == 1:
            for X_spa, y in train_iter:
                X_spa, y = X_spa.to(device), y.to(device)
                y_pred, logits_dict = net(X_spa,train=True)
                ls_joint = loss(y_pred, y.long())

                loss_single = 0
                for key in logits_dict.keys():
                    loss_single = loss_single + loss(logits_dict[key], y.long()) * (1 / len(logits_dict.keys()))

                ls = ls_joint + loss_single



                optimizer.zero_grad()
                ls.backward()
                optimizer.step()

                train_loss_sum += ls.cpu().item()
                train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                trained_samples_counter += y.shape[0]
                batch_counter += 1
                epoch_first_iter = 0

        elif model_type_flag == 2:
            for X_spe, y in train_iter:
                X_spe, y = X_spe.to(device), y.to(device)
                y_pred = net(X_spe)

                ls = loss(y_pred, y.long())

                optimizer.zero_grad()
                ls.backward()
                optimizer.step()

                train_loss_sum += ls.cpu().item()
                train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                trained_samples_counter += y.shape[0]
                batch_counter += 1
                epoch_first_iter = 0
        elif model_type_flag == 3:
            for X_spa, X_spe, y in train_iter:
                X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)

                y_pred = net(X_spa, X_spe)

                ls = loss(y_pred, y.long())

                optimizer.zero_grad()
                ls.backward()
                optimizer.step()

                train_loss_sum += ls.cpu().item()
                train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                trained_samples_counter += y.shape[0]
                batch_counter += 1
                epoch_first_iter = 0

        val_acc, val_loss = evaluation.evaluate_OA(val_iter, net, loss, device, model_type_flag)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), save_weight_path)

        torch.cuda.empty_cache()

        train_loss_list.append(train_loss_sum)
        train_acc_list.append(train_acc_sum / trained_samples_counter)

        print('epoch: %d, train loss: %.6f' % (epoch + 1, train_loss_sum / batch_counter))

    pred_test = []
    with torch.no_grad():
        data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)
        net.load_state_dict(torch.load(save_weight_path))
        net.eval()
        train_acc_sum, samples_num_counter = 0.0, 0
        if model_type_flag == 1:  # data for single spatial net
            for index, y in test_iter:
                index = list(index.numpy())
                X_spa = HSI_create_pathes(data_padded_torch, height, width, index, patch_length, 1)
                X_spa = X_spa.to(device)
                y = y.to(device)
                if model_3D_spa_flag == 1:
                    X_spa = X_spa.unsqueeze(1)

                tic2 = time.perf_counter()
                y_pred = net(X_spa)
                toc2 = time.perf_counter()

                pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
        elif model_type_flag == 2:  # data for single spectral net
            for index, y in test_iter:
                index = list(index.numpy())
                X_spe = HSI_create_pathes(data_padded_torch, height, width, index, patch_length, 2)
                X_spe = X_spe.to(device)
                y = y.to(device)

                tic2 = time.perf_counter()
                y_pred = net(X_spe)
                toc2 = time.perf_counter()

                pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
        elif model_type_flag == 3:  # data for spectral-spatial net
            for index, y in test_iter:
                index = list(index.numpy())
                X_spa = HSI_create_pathes(data_padded_torch, height, width, index, patch_length, 1)
                X_spe = HSI_create_pathes(data_padded_torch, height, width, index, patch_length, 2)
                X_spa = X_spa.to(device)
                X_spe = X_spe.to(device)
                y = y.to(device)

                tic2 = time.perf_counter()
                y_pred = net(X_spa, X_spe)
                toc2 = time.perf_counter()

                pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))

        y_gt = gt_reshape[test_data_index] - 1
        OA = metrics.accuracy_score(y_gt, pred_test)
        confusion_matrix = metrics.confusion_matrix(pred_test, y_gt)
        ECA, AA = evaluation.AA_ECA(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, y_gt)


        str_results = '\n======================' \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA) \
                      + "\nAA=" + str(AA) \
                      + '\nkpp=' + str(kappa) \
                      + '\nacc per class:' + str(ECA) \

        print(str_results)