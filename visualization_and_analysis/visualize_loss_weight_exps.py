import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

alpha = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
beta = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
gamma = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

res_path = 'C:/Users/lsc/Desktop/exps_results/SHL_loss_weight_torch_exps'

n_exp_repeat = 11
total_accs = []
for i in range(n_exp_repeat):
    accs = []
    for a, b, g in zip(alpha, beta, gamma):
        exp_path = os.path.join(res_path, f'SHL_loss{a},{b},{g}_exp{i}')
        log_path = os.path.join(exp_path, 'classification_results.txt')
        lines = open(log_path, encoding='utf-8').readlines()
        acc = float(lines[-2][36:43])  # Penultimate line, and only cut acc value
        accs.append(acc)
    total_accs.append(accs)
total_accs = np.array(total_accs)

total_train_times = []
for i in range(n_exp_repeat):
    train_times = []
    for a, b, g in zip(alpha, beta, gamma):
        exp_path = os.path.join(res_path, f'SHL_loss{a},{b},{g}_exp{i}')
        log_path = os.path.join(exp_path, 'log.txt')
        lines = open(log_path, encoding='utf-8').readlines()
        for line in lines:
            if re.search(r'END JOINT', line):
                # print(line)
                joint_train_time = float(line[-8:-2])
                train_times.append(joint_train_time)
                # print(joint_train_time)
                break

    total_train_times.append(train_times)
total_train_times = np.array(total_train_times)


mean_acc = np.mean(total_accs, axis=0)
mean_time = np.mean(total_train_times, axis=0)

poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=4)),
    ("std_scalar", StandardScaler()),
    ("lr", LinearRegression())
])
x = beta
X = np.reshape(x, (-1, 1))
y = mean_acc
poly_pipeline.fit(X, y)
y_predict_pipeline = poly_pipeline.predict(X)
print(x[np.argmax(y)])
print(x[np.argmax(y_predict_pipeline)])
mse = mean_squared_error(y, y_predict_pipeline)
print(f'mse:{mse}')

plt.scatter(x, y)
# plt.plot(np.sort(x), y_predict_pipeline[np.argsort(x)], color='r')
plt.plot(x, y_predict_pipeline, color='r')
plt.show()


def show_bar_chart():
    plt.figure()
    x = beta
    # create an index for each tick position
    xi = list(range(len(x)))
    y = mean_acc

    axes = plt.gca()
    # axes.set_xlim([min(x),max(x)])
    axes.set_ylim([min(y) - 0.003, max(y) + .001])

    # plot the index for the x-values
    bars = plt.bar(xi, y, )
    plt.xlabel('$\\beta$')
    plt.ylabel('accuracy')
    plt.xticks(xi, x, fontsize=8, rotation=60)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.2, yval, '%.5f' % yval, va='top', fontsize=8, rotation=90, color='w')
    bars[np.argmax(y)].set_color('r')
    # plt.legend()
    plt.show()

def show_train_time_bar_chart():
    plt.figure()
    x = beta
    # create an index for each tick position
    xi = list(range(len(x)))
    y = mean_time

    axes = plt.gca()
    # axes.set_xlim([min(x),max(x)])
    axes.set_ylim([min(y) - 0.003, max(y) + .001])

    # plot the index for the x-values
    bars = plt.bar(xi, y, )
    plt.xlabel('$\\beta$')
    plt.ylabel('training time')
    plt.xticks(xi, x, fontsize=8, rotation=60)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.2, yval, '%.5f' % yval, va='top', fontsize=8, rotation=90, color='w')
    bars[np.argmax(y)].set_color('r')
    # plt.legend()
    plt.show()


show_bar_chart()
show_train_time_bar_chart()
