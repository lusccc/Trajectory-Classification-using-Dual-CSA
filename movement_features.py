import numpy as np

trjs = np.load('./geolife/trjs.npy', allow_pickle=True)
labels = np.load('./geolife/labels.npy')
label = np.unique(labels)
print(1)