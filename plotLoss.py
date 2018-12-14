import numpy as np
import matplotlib.pyplot as plt

X = []
Y_6block = []
Y_9block = []

with open("Results/loss/loss_6.txt", "r") as f:
    content = f.readlines()
    for line in content:
        ind = [9, 11, 13, 17, 19, 21]
        losses = np.array([float(line.split()[i]) for i in ind])
        losses = sum(losses)
        e = float(line.split()[1].rstrip(','))
        if e < 150:
            it = float(line.split()[3].rstrip(','))
            epoch = e + it/550.
            X.append(epoch)
            Y_6block.append(losses)

X2 = []

with open("Results/loss/loss_9.txt", "r") as f:
    content = f.readlines()
    for line in content:
        ind = [9, 11, 13, 17, 19, 21]
        losses = np.array([float(line.split()[i]) for i in ind])
        losses = sum(losses)
        e = float(line.split()[1].rstrip(','))
        if e < 150:
            it = float(line.split()[3].rstrip(','))
            epoch = e + it/550.
            X2.append(epoch)
            Y_9block.append(losses)

plt.plot(X, Y_6block, color = "red", linewidth = 0.8)
plt.plot(X2, Y_9block, linewidth = 0.8)
plt.xlabel("Epoch")
plt.ylabel("Objective Loss")
plt.legend(["ResNet-9", "ResNet-6"])
plt.show()

print(np.average(Y_6block))
print(np.average(Y_9block))
