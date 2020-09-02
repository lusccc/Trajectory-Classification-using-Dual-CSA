import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

dim_limit = 408

dim = [i for i in range(8, dim_limit, 8)]

res_path = 'C:/Users/lsc/Desktop/exp_res_SHL_all'


def read_res_acc(folder_prefix):
    accs = []
    for i in range(8, dim_limit, 8):
        exp_path = os.path.join(res_path, f'{folder_prefix}{i}')
        log_path = os.path.join(exp_path, 'log.txt')
        log = open(log_path, encoding='utf-8').readlines()
        acc = float(log[-2][57:])  # Penultimate line, and only cut acc value
        accs.append(acc)
    return accs


acc_exp1 = read_res_acc('exp1st_SHL_t7s3e')
acc_exp2 = read_res_acc('exp2nd_SHL_t7s3e')
acc_exp3 = read_res_acc('exp3rd_SHL_t7s3e')
acc_exp4 = read_res_acc('exp4th_SHL_t7s3e')
acc_exp5 = read_res_acc('exp5th_SHL_t7s3e')
acc_exp6 = read_res_acc('exp6th_SHL_t7s3e')
acc_exp7 = read_res_acc('exp7th_SHL_t7s3e')
acc_exp8 = read_res_acc('exp8th_SHL_t7s3e')
acc_exp9 = read_res_acc('exp9th_SHL_t7s3e')
acc_exp10 = read_res_acc('exp10th_SHL_t7s3e')

all_acc = np.array(
    [acc_exp1, acc_exp2, acc_exp3, acc_exp4,
     acc_exp5, acc_exp6, acc_exp7, acc_exp8,  acc_exp10]
)
mean_acc = np.mean(all_acc, axis=0)

poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=4)),
    ("std_scalar", StandardScaler()),
    ("lr", LinearRegression())
])
x = dim
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
    x = dim
    # create an index for each tick position
    xi = list(range(len(x)))
    y = mean_acc

    axes = plt.gca()
    # axes.set_xlim([min(x),max(x)])
    axes.set_ylim([min(y) - 0.003, max(y) + .001])

    # plot the index for the x-values
    bars = plt.bar(xi, y, )
    plt.xlabel('embedding dimension in latent space')
    plt.ylabel('accuracy')
    plt.xticks(xi, x, fontsize=8, rotation=60)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.2, yval, '%.5f' % yval, va='top', fontsize=8, rotation=90, color='w')
    bars[np.argmax(y)].set_color('r')
    # plt.legend()
    plt.show()


show_bar_chart()
