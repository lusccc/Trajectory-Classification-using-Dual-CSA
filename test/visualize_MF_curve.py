import numpy as np
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler, StandardScaler

labels = np.load('../data/geolife_features/trjs_segs_features_labels.npy')[:1000]

trjs_segs_features = np.load('../data/geolife_features/trjs_segs_features.npy')[:1000]
trjs_segs_features = np.squeeze(trjs_segs_features)
data = trjs_segs_features[:, :, 1]

idx = [127,14,80,110,159,295,240,289]

for i in idx:
    plt.figure()
    v = data[i]
    print(v.shape)
    plt.plot(v)
    plt.savefig('MF_curve_imgs/%d.png' % i)