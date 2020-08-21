import matplotlib.pyplot as plt
import numpy as np
dataset = 'Trajectory_Feature_Dataset'
# dataset_name = 'geolife'
train_segs_noise_features = np.load(f'../data/{dataset}_features/trjs_segs_features_train_noscale.npy')
labels = np.load(f'../data/{dataset}_features/trjs_segs_features_labels_train.npy')
train_segs_noise_features = np.squeeze(train_segs_noise_features)
print()
segs_MF = []
for i, label in enumerate(labels):
    label = np.argmax(label, )
    # 0,    1,    2,   3,         4           5
    # walk, bike, bus, driving, train/subway, run
    # modes_to_use = [0, 1, 2, 3, 4]
    if label == 2:
        #                   0  1         2    3  4
        # delta_t, hour, d, v, a, h, hc, hcr, s, tn
        MFs = train_segs_noise_features[i][:, 3]
        segs_MF.extend(MFs)

segs_MF = np.array(segs_MF)
print(segs_MF.shape)
plt.hist(segs_MF, 300)
plt.show()
