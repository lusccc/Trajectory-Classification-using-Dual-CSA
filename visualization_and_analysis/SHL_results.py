import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


res_path = 'C:/Users/lsc/Desktop/exp_res_SHL_all'


def read_res_acc(path):
    accs = []
    log_path = os.path.join(os.path.join(res_path, path), 'log.txt')
    log = open(log_path, encoding='utf-8').readlines()
    acc = float(log[-19][57:])  # Penultimate line, and only cut acc value
    accs.append(acc)
    return accs


all_acc = []
for i in range(10):
    acc_exp = read_res_acc(f'exp{i}_SHL_t7s3e304a1b3y1_optimal')
    all_acc.append(acc_exp)
all_acc = np.array(all_acc)

mean_acc = np.mean(all_acc, axis=0)

print() #.88190
