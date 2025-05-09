import matplotlib.pyplot as plt
import numpy as np

data1 = np.loadtxt("ex3/data/data1.csv", skiprows=1, delimiter=",")

plt.scatter(data1[:, 0], data1[:, 1])
plt.savefig("ex3/i_tzimas/data1_scatter.png")
