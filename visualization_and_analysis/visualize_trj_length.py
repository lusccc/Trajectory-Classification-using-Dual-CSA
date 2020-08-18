import matplotlib.pyplot as plt
import numpy as np

# from params import MIN_N_POINTS

# trjs = np.load('../data/geolife_extracted/trjs_train.npy', allow_pickle=True)
trjs = np.load('../data/SHL_extracted/trjs_train.npy', allow_pickle=True)
lens = []
for trj in trjs:
    l = len(trj)
    if l > 10:
        lens.append(l)
lens = np.array(lens)
print(lens.mean())  # 597.9427522436769 for geolife
plt.hist(lens, 300)
plt.show()
