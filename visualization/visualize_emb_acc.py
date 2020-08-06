import matplotlib.pyplot as plt

dim = [i for i in range(8, 200, 8)]

# acc = [.87162, .88477, .88289, .88271, .88195, .88045, .87914, .88346, .88120, .88102, .87932, .88684, .88402, .88383,
#        .88233, .88647, .88383, .87820, .88195, .88045, .88064, .88496, .88139, .88139]

acc = [0.88215, 0.88690]

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
    plt.text(bar.get_x() + 0.2, yval, '%.5f' % yval, va='top', fontsize=8, rotation=90, color='w')
bars[11].set_color('r')
# plt.legend()
plt.show()
