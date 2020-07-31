import matplotlib.pyplot as plt

dim = [i for i in range(8, 136, 8)]

acc1 = [0.87593, 0.88089, 0.88213, 0.87965, 0.88834, 0.89082, 0.88586, 0.88213, 0.88213, 0.88337, 0.88710, 0.88710, 0.88337, 0.88834, 0.88710, 0.88213]
acc2 = [0.87717, 0.88586, 0.88089, 0.87469, 0.88213, 0.88586, 0.88586, 0.88337, 0.88213, 0.88213, 0.88213, 0.88586, 0.88213, 0.88337, 0.88213, 0.88337]
acc3 =
plt.figure()


x = dim
# create an index for each tick position
xi = list(range(len(x)))
y = acc

axes = plt.gca()
# axes.set_xlim([min(x),max(x)])
axes.set_ylim([min(y) - 0.003, max(y) + .001])

# plot the index for the x-values
bars = plt.bar(xi, y, )
plt.xlabel('embedding dimension in latent space')
plt.ylabel('accuracy')
plt.xticks(xi, x, fontsize=8, rotation=60)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+0.2, yval,  '%.5f' % yval, va='top', fontsize=8, rotation=90, color='w')
bars[11].set_color('r')
# plt.legend()
plt.show()
