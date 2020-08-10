import numpy as np
import matplotlib.pyplot as plt


dataset = 'SHL'

x_features_series_train = np.load(f'../data/{dataset}_features/trjs_segs_features_train.npy', )
x_features_series_test =  np.load(f'../data/{dataset}_features/trjs_segs_features_test.npy', )
y_train =                 np.load(f'../data/{dataset}_features/trjs_segs_features_labels_train.npy', )
y_test =                  np.load(f'../data/{dataset}_features/trjs_segs_features_labels_test.npy', )

y = np.vstack([y_test, y_train])
y = np.argmax(y, 1)
unique, n_each = np.unique(y, return_counts=True)

print()

def show_bar_chart():
    plt.figure()
    x = ['walk', 'bike', 'bus', 'driving', 'train']
    # create an index for each tick position
    xi = list(range(len(x)))
    y = n_each

    axes = plt.gca()
    # axes.set_xlim([min(x),max(x)])
    # axes.set_ylim([min(y) - 0.003, max(y) + .001])

    # plot the index for the x-values
    bars = plt.bar(xi, y, )
    plt.xlabel('embedding dimension in latent space')
    plt.ylabel('accuracy')
    plt.xticks(xi, x, fontsize=8, rotation=60)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.2, yval, '%.5f' % yval, va='top', fontsize=8, rotation=90, color='w')
    # bars[np.argmax(y)].set_color('r')
    # plt.legend()
    plt.show()

show_bar_chart()