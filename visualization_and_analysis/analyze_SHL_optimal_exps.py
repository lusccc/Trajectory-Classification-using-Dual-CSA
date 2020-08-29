import os
import re

import matplotlib.pyplot as plt
import numpy as np

res_path = 'C:/Users/lsc/Desktop/exps_results/SHL_optimal_exps'
total_accs = []
for i in range(10):
    exp_path = os.path.join(res_path, f'SHL_optimal_exp{i}')
    log_path = os.path.join(exp_path, 'classification_results.txt')
    lines = open(log_path, encoding='utf-8').readlines()
    acc = float(lines[-2][36:43])  # Penultimate line, and only cut acc value
    total_accs.append(acc)
mean_acc = np.mean(total_accs, axis=0)
print(mean_acc) #0.887856