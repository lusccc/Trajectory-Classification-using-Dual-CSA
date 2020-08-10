import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

alpha = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
beta = (.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
gamma = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

res_path = 'C:/Users/lsc/Desktop/exp_res_SHL_all'


def read_res_acc(folder_prefix):
    accs = []
    for al, be, ga in zip(alpha, beta, gamma):
        exp_path = os.path.join(res_path, f'{folder_prefix}a{al}b{be}y{ga}')
        log_path = os.path.join(exp_path, 'log.txt')
        log = open(log_path, encoding='utf-8').readlines()
        acc = float(log[-2][57:])  # Penultimate line, and only cut acc value
        accs.append(acc)
    return accs


all_acc = []
for i in range(10):
    acc_exp = read_res_acc(f'exp{i}_SHL_t7s3e304')
    all_acc.append(acc_exp)
all_acc = np.array(all_acc)


mean_acc = np.mean(all_acc, axis=0)

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


show_bar_chart()