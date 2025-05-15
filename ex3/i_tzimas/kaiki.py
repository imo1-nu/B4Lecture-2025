"""Solution to part 1 of ex3 on linear regression."""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso


def ols(
    data: np.ndarray, order: int, reg: str = None, alpha: float = 1.0
) -> np.ndarray:
    """Ordinary least squares implementation with regularization options.

    Params:
    Returns:
    """
    y = data[:, -1][:, np.newaxis]
    x = data[:, :-1]
    # Increase order with exponent terms
    x_raw = np.hstack([x**i for i in range(1, order + 1)])
    # ---------------------------------------------
    w_sk = Lasso(alpha).fit(x, y)
    print(w_sk.intercept_, w_sk.coef_)
    # ---------------------------------------------
    x = np.hstack((np.ones(len(x_raw))[:, np.newaxis], x_raw))

    match reg:
        case "lasso":
            w = np.random.uniform(-1, 1, x.shape[1])

            for _ in range(1000):
                w_prev = np.copy(w)
                w[0] = np.sum(y - x[:, 1:] @ w[1:]) / len(x)

                for i in range(1, len(w)):
                    x_i = x[:, i]
                    r = (y - np.delete(x, i, 1) @ np.delete(w, i)) @ x_i
                    if r < -alpha:
                        w[i] = r + alpha / np.sum(x_i**2)
                    elif r > alpha:
                        w[i] = r - alpha / np.sum(x_i**2)
                    else:
                        w[i] = 0

                if np.linalg.norm(w - w_prev, 2) < 10e-5:
                    break
        case "ridge":
            w = np.dot(
                np.dot(
                    np.linalg.inv(np.dot(x.T, x) + alpha * np.identity(x.shape[1])), x.T
                ),
                y,
            )
        case "elastic_net":
            w = 0
        case _:
            w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

    return w


def main() -> None:
    """Main routine.

    Params: None
    Returns: None
    """
    # Linear regression with OLS

    # Data 1
    data1 = np.loadtxt("ex3/data/data1.csv", skiprows=1, delimiter=",")
    w1 = ols(data1, 1, "lasso", 0.1)

    print(w1)

    # Plots
    plt.scatter(data1[:, 0], data1[:, 1], 20, "#912583", zorder=2)  # Data points
    plt.title("Data 1 Scatterplot")
    plt.grid()
    plt.savefig("ex3/i_tzimas/data1_scatter.png")
    x = np.array([min(data1[:, 0]), max(data1[:, 0])])
    plt.plot(x, w1[0] + w1[1] * x)
    plt.show()
    plt.clf()

    # Data 2
    data2 = np.loadtxt("ex3/data/data2.csv", skiprows=1, delimiter=",")
    w2 = ols(data2, 2)

    print(w2)

    # Plots
    plt.scatter(data2[:, 0], data2[:, 1], 20, "#912583", zorder=2)
    plt.title("Data 2 Scatterplot")
    plt.grid()
    plt.savefig("ex3/i_tzimas/data2_scatter.png")
    plt.clf()

    # Data 3
    data3 = np.loadtxt("ex3/data/data3.csv", skiprows=1, delimiter=",")
    w3 = ols(data3, 3)

    print(w3)

    # Plots
    plt.subplot(111, projection="3d").scatter(data3[:, 0], data3[:, 1], data3[:, 2])
    plt.title("Data 3 Scatterplot")


if __name__ == "__main__":
    main()
