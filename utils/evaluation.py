import torch
import numpy as np

from sklearn import metrics
from operator import truediv


def evaluate_OA(data_iter, net, loss, device, model_type_flag):
    acc_sum, samples_counter = 0, 0

    with torch.no_grad():
        net.eval()
        if model_type_flag == 1:  # data for single spatial net
            for X_spa, y in data_iter:
                loss_sum = 0
                X_spa, y = X_spa.to(device), y.to(device)
                y_pred = net(X_spa)
                ls = loss(y_pred, y.long())

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]
        elif model_type_flag == 2:  # data for single spectral net
            for X_spe, y in data_iter:
                loss_sum = 0
                X_spe, y = X_spe.to(device), y.to(device)
                data_dict = net(X_spe)
                y_pred = data_dict['logits']
                ls = loss(y_pred, y.long())

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]
        elif model_type_flag == 3:  # data for spectral-spatial net
            for X_spa, X_spe, y in data_iter:
                loss_sum = 0
                X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                y_pred = net(X_spa, X_spe)
                ls = loss(y_pred, y.long())

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]

    return [acc_sum / samples_counter, loss_sum]


def AA_ECA(confusion_matrix):
    diag_list = np.diag(confusion_matrix)
    row_sum_list = np.sum(confusion_matrix, axis=1)
    each_per_acc = np.nan_to_num(truediv(diag_list, row_sum_list))
    avg_acc = np.mean(each_per_acc)

    return each_per_acc, avg_acc



