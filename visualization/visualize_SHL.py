import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
train_segs_features = np.load('../data/SHL_features/trjs_segs_features_train_noscale.npy')
labels = np.load('../data/SHL_features/trjs_segs_features_labels_train.npy')
train_segs_features = np.squeeze(train_segs_features)
print()
segs_hcrs = []
for i, label in enumerate(labels):
    label = np.argmax(label, )
    if label == 1:
        hcrs = train_segs_features[i][:, 2]
        segs_hcrs.extend(hcrs)

segs_hcrs = np.array(segs_hcrs)
print()
plt.hist(segs_hcrs, 300)
plt.show()