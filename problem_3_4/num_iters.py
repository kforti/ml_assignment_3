import matplotlib.pyplot as plt

iters = []
times = []
accuracies = []

with open("num_iterations_8100_samples", "r") as f:
    for line in f:
        items = line.split(',')
        iters.append(int(items[0]))
        times.append(float(items[1]))
        accuracies.append(float(items[2]))

fig, ax = plt.subplots()
ax.plot(iters, accuracies)

ax.set(xlabel='Iterations', ylabel='accuracy',
       title='Iterations vs Accuracy')
ax.grid()

fig.savefig("I_vs_A.png")
plt.show()

fig, ax = plt.subplots()
ax.plot(iters, times)

ax.set(xlabel='Iterations', ylabel='Times (s)',
       title='Iterations vs Time')
ax.grid()

fig.savefig("I_vs_T.png")
plt.show()
