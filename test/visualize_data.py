import matplotlib.pyplot as plt
import numpy as np

features_segments = np.load('./geolife/features_segments.npy')[:]
features_segments_labels = np.load('./geolife/features_segments_labels.npy')[:]

fs = features_segments[:,:,0].flatten()
plt.hist(fs, bins='auto')
plt.show()