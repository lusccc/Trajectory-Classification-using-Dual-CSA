import matplotlib.pyplot as plt

dim = [5, 5, 6, 8, 9, 9, 9, 9]
tau = [6, 8, 8, 8, 6, 7, 8, 9]

acc = [.88308, .88252, .88120, .88684, .88477, .88195, .88195, .88177]

plt.figure()
plt.plot( acc)
for a, b, c in zip(dim, tau, acc):
    plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)

# plt.legend()
plt.savefig("dimtau_acc.png")
