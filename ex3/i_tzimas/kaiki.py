import matplotlib.pyplot as plt
import numpy as np


def ols(data: np.ndarray, order: int) -> np.ndarray:
    y = data[:, -1][:, np.newaxis]
    x = data[:, :-1]
    x = np.hstack([x**i for i in range(1, order + 1)])
    x = np.hstack((np.ones(len(x))[:, np.newaxis], x))

    w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    print(w)
    return w


def main() -> None:
    # Scatterplots
    data1 = np.loadtxt("ex3/data/data1.csv", skiprows=1, delimiter=",")

    plt.scatter(data1[:, 0], data1[:, 1], 20, "#912583", zorder=2)
    plt.title("Data 1 Scatterplot")
    plt.grid()
    plt.savefig("ex3/i_tzimas/data1_scatter.png")
    plt.clf()

    data2 = np.loadtxt("ex3/data/data2.csv", skiprows=1, delimiter=",")

    plt.scatter(data2[:, 0], data2[:, 1], 20, "#912583", zorder=2)
    plt.title("Data 2 Scatterplot")
    plt.grid()
    plt.savefig("ex3/i_tzimas/data2_scatter.png")
    plt.clf()

    data3 = np.loadtxt("ex3/data/data3.csv", skiprows=1, delimiter=",")
    plt.subplot(111, projection="3d").scatter(data3[:, 0], data3[:, 1], data3[:, 2])
    plt.title("Data 3 Scatterplot")
    # plt.show()

    # Linear regression with OLS

    # Data 1
    w1 = ols(data1, 1)
    # TODO: Scatterplot

    # Data 2
    w2 = ols(data2, 2)
    # TODO: Scatterplot

    # Data 3
    w3 = ols(data3, 3)
    # TODO: Scatterplot


if __name__ == "__main__":
    main()
