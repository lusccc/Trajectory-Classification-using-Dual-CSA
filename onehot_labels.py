import numpy as np

labels = np.load('./geolife/features_segments_labels.npy')
labels_unique = np.unique(labels)
print(labels_unique)
n_class = len(labels_unique)
onehot_labels = []
for label in labels:
    onehot = np.zeros((n_class,))
    onehot[label] = 1
    onehot_labels.append(onehot)
onehot_labels = np.array(onehot_labels)
np.save('./geolife/features_segments_onehot_labels.npy', onehot_labels)
