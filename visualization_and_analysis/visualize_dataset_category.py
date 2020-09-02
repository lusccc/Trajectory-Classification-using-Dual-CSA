import numpy as np
import matplotlib.pyplot as plt

dataset = 'geolife'


def count_number_each_class(dataset):
    # x_features_series_train = np.load(f'../data/{dataset}_features/multi_feature_segs_train.npy', )
    # x_features_series_test = np.load(f'../data/{dataset}_features/multi_feature_segs_test.npy', )
    y_train = np.load(f'../data/{dataset}_features/multi_feature_seg_labels_test.npy', )
    y_test = np.load(f'../data/{dataset}_features/multi_feature_seg_labels_train.npy', )

    y = np.vstack([y_test, y_train])
    y = np.argmax(y, 1)
    unique, n_each = np.unique(y, return_counts=True)
    return n_each


plt.figure()
x = ['walk', 'bike', 'bus', 'driving', 'train']
# create an index for each tick position
xi = np.array(list(range(len(x))))

axes = plt.gca()
# axes.set_xlim([min(x),max(x)])
# axes.set_ylim([min(y) - 0.003, max(y) + .001])

y1 = count_number_each_class('geolife')
y2 = count_number_each_class('SHL')
# plot the index for the x-values
bars1 = plt.bar(xi - 0.2, y1, width=0.4, label='Geolife' )
bars2 = plt.bar(xi + 0.2, y2, width=0.4, label='SHL')
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x()+.025, yval + 300, '%d' % yval, va='top', fontsize=8, )
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x()+.035, yval + 300, '%d' % yval, va='top', fontsize=8, )
plt.xlabel('transportation mode')
plt.ylabel('number of segments')
plt.xticks(xi, x, fontsize=8, )
plt.legend()
plt.show()
