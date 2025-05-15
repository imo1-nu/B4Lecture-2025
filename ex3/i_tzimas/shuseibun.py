import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca(data, output_size=None, use_percent=False):
    # Subtract mean for standardization
    data_std = data - np.mean(data, axis=0)
    # Covariate matrix
    data_cov = data_std.T @ data_std / len(data_std)
    # Calculate eigenvalues/vectors
    eig_val, eig_vec = np.linalg.eig(data_cov)
    # Sort based on eigenvalues in descending order (large value = important component)
    sort = np.argsort(eig_val)
    eig_val = eig_val[sort[::-1]]
    pc = eig_vec[:, sort[::-1]]

    if not use_percent:
        # Return all PC or the requested size
        if output_size:
            return pc[:output_size]
        else:
            return pc
    else:
        # Return as many as needed for 90% explained variance
        size = 1
        while np.cumsum([i / np.sum(eig_val) for i in eig_val[:size]]) <= 0.9:
            size += 1
        return pc[:size]


def main():
    # Data 1
    data1 = np.loadtxt("ex3/data/data1.csv", skiprows=1, delimiter=",")
    data1_std = data1 - np.mean(data1, axis=0)
    data1_pc = pca(data1)

    # Plot
    plt.scatter(
        data1_std[:, 0], data1_std[:, 1], 20, "#912583", zorder=2
    )  # Data points
    plt.title("Data 1 Scatterplot with PCA")
    plt.grid()
    for v in data1_pc:
        plt.axline((0, 0), (v[0], v[1]), ls="--")
    plt.savefig("ex3/i_tzimas/data1_pca.png")
    plt.clf()

    # Data 2
    data2 = np.loadtxt("ex3/data/data2.csv", skiprows=1, delimiter=",")
    data2_std = data2 - np.mean(data2, axis=0)
    data2_pc = pca(data2)

    # Plot
    plt.scatter(
        data2_std[:, 0], data2_std[:, 1], 20, "#912583", zorder=2
    )  # Data points
    plt.title("Data 2 Scatterplot with PCA")
    plt.grid()
    for v in data2_pc:
        plt.axline((0, 0), (v[0], v[1]), ls="--")
    plt.savefig("ex3/i_tzimas/data2_pca.png")
    plt.clf()

    # Data 3
    data3 = np.loadtxt("ex3/data/data3.csv", skiprows=1, delimiter=",")
    data3_std = data3 - np.mean(data3, axis=0)
    data3_pc = pca(data3)

    # Plot
    plt.subplot(111, projection="3d").scatter(
        data3_std[:, 0], data3_std[:, 1], data3_std[:, 2]
    )
    plt.title("Data 3 Scatterplot with PCA")
    for v in data3_pc:
        plt.plot([0, v[0]], [0, v[1]], [0, v[2]])
    plt.show()


if __name__ == "__main__":
    main()
