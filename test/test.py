import matplotlib.pyplot as plt
import numpy as np

dts = np.load('./geolife/train_mf_segments.npy')
stss = dts[:,0,:,9].reshape((-1, 1))
plt.plot(stss)
plt.show()
NNN=1
cc = []
def c():
    a = 1
    NNN += 1
    cc.append()

if __name__ == '__main__':
    NNN = 0





