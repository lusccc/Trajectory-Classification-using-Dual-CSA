import numpy as np
import matplotlib.pyplot as plt
drop_percentage = list(range(0, 90,10))
acc_geolife = [0.89339, 0.86911, 0.85389, 0.85213, 0.83947, 0.81456, 0.79699, 0.77400, 0.74048]
acc_SHL = [0.89236, 0.89014, 0.87500, 0.80669, 0.75464, 0.71564, 0.68802, 0.62951, 0.61765]



axes = plt.gca()
axes.set_ylim([0.6, 0.9])

plt.scatter(drop_percentage, acc_geolife, label='Geolife')
plt.plot(drop_percentage, acc_geolife)
plt.scatter(drop_percentage, acc_SHL, label='SHL')
plt.plot(drop_percentage, acc_SHL)
plt.xlabel('percentage of discarded points in the trajectory')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()
