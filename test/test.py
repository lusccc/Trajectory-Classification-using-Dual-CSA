
import matplotlib.pyplot as plt
x = [0.00001,0.001,0.01,0.1,0.5,1,5]
# create an index for each tick position
xi = list(range(len(x)))
y = [0.945,0.885,0.893,0.9,0.996,1.25,1.19]
plt.ylim(0.8,1.4)
# plot the index for the x-values
plt.plot(xi, y, marker='o', linestyle='--', color='r', label='Square')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(xi, x)
plt.title('compare')
plt.legend()
plt.show()
