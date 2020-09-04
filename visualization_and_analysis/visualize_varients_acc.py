import numpy as np
import matplotlib.pyplot as plt

plt.figure()
x = ['Dual-CA-Softmax', 'CSA-RP', 'CSA-FS', 'Dual-CSA', ]
# create an index for each tick position
xi = np.array(list(range(len(x))))

axes = plt.gca()
# axes.set_xlim([min(x),max(x)])

#acc
y1 = [0.85854, 0.86516, 0.86172, 0.89339]
y2 = [0.87309, 0.87951, 0.87003, 0.89236]
#f1
# y1 = [0.85801, 0.86383, 0.85958, 0.89243]
# y2 = [0.87301, 0.88012, 0.86906, 0.89337]
axes.set_ylim([0.8, 0.9])
# plot the index for the x-values
bars1 = plt.bar(xi - 0.2, y1, width=0.4, label='Geolife')
bars2 = plt.bar(xi + 0.2, y2, width=0.4, label='SHL')
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .0025, '%.5f' % yval, va='top', fontsize=8, )
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .0025, '%.5f' % yval, va='top', fontsize=8, )
plt.xlabel('variants/our network')
plt.ylabel('accuracy')
# plt.ylabel('F1 score')
plt.xticks(xi, x, fontsize=8, )
plt.legend()
plt.show()
