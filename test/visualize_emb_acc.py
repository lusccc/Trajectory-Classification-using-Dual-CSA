import matplotlib.pyplot as plt

dim = [i for i in range(8, 200, 8)]

acc = [.87162, .88477, .88289, .88271, .88195, .88045, .87914, .88346, .88120, .88102, .87932, .88684, .88402, .88383,
       .88233, .88647, .88383, .87820, .88195, .88045, .88064, .88496, .88139, .88139]

plt.figure()
plt.plot(dim,acc)
for a, b in zip(dim, acc):
       plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)

# plt.legend()
plt.savefig("acc.png")
