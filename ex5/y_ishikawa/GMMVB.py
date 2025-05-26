"""Apply GMMVB to data and visualize."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal

DIMENSION_1D = 1
DIMENSION_2D = 2


class GMMVB:
    """Variational Bayesian Gaussian Mixture Model (VB-GMM).

    This implementation is based on:
    https://github.com/beginaid/GMM-EM-VB

    Parameters
    ----------
    K : int
        The number of clusters.

    Attributes
    ----------
    K : int
        Number of clusters.
    eps : float
        A small constant to prevent numerical instabilities.
    N : int
        Number of samples (set during initialization).
    D : int
        Dimensionality of the data (set during initialization).
    alpha, beta, nu, m, W : np.ndarray
        Variational parameters for each component.
    r : np.ndarray
        Responsibilities for each data point.
    """

    def __init__(self, K):
        """Constructor.

        Args:
            K (int): The number of clusters.

        Returns:
            None.

        Note:
            eps (float): Small amounts to prevent overflow and underflow.
        """
        self.K = K
        self.eps = np.spacing(1)

    def init_params(self, X):
        """Initialize the parameters.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # The size of X is (N, D)
        self.N, self.D = X.shape
        # Set the initial weighting factors
        self.alpha0 = 0.01
        self.beta0 = 1.0
        self.nu0 = float(self.D)
        self.m0 = np.random.randn(self.D)
        self.W0 = np.eye(self.D)
        self.r = np.random.randn(self.N, self.K)
        # Keep the updated parameters in the field
        self.alpha = np.ones(self.K) * self.alpha0
        self.beta = np.ones(self.K) * self.beta0
        self.nu = np.ones(self.K) * self.nu0
        self.m = np.random.randn(self.K, self.D)
        self.W = np.tile(self.W0[None, :, :], (self.K, 1, 1))

    def gmm_pdf(self, X):
        """Calculate the log-likelihood of the D-dimensional mixed Gaussian distribution at N data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            Probability density function (numpy ndarray): Values of the mixed D-dimensional Gaussian distribution at N data whose size is (N, K).
        """
        pi = self.alpha / (np.sum(self.alpha, keepdims=True) + np.spacing(1))  # (K)
        return np.array(
            [
                pi[k]
                * multivariate_normal.pdf(
                    X, mean=self.m[k], cov=la.pinv(self.nu[:, None, None] * self.W)[k]
                )
                for k in range(self.K)
            ]
        ).T  # (N, K)

    def e_step(self, X):
        """Execute the variational E-step of VB.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Calculate and update the optimized Pi
        log_pi_tilde = digamma(self.alpha) - digamma(self.alpha.sum())  # (K)
        # Calculate the optimized Lambda_titlde
        log_sigma_tilde = np.sum(
            [digamma((self.nu + 1 - i) / 2) for i in range(self.D)]
        ) + (self.D * np.log(2) + (np.log(la.det(self.W) + np.spacing(1))))  # (K)
        nu_tile = np.tile(self.nu[None, :], (self.N, 1))  # (N, K)
        res_error = np.tile(X[:, None, None, :], (1, self.K, 1, 1)) - np.tile(
            self.m[None, :, None, :], (self.N, 1, 1, 1)
        )  # (N, K, 1, D)
        quadratic = (
            nu_tile
            * (
                (res_error @ np.tile(self.W[None, :, :, :], (self.N, 1, 1, 1)))
                @ res_error.transpose(0, 1, 3, 2)
            )[:, :, 0, 0]
        )  # (N, K)
        log_rho = (
            (log_pi_tilde[None, :])
            + (0.5 * log_sigma_tilde)[None, :]
            - (0.5 * self.D / (self.beta + np.spacing(1)))[None, :]
            - (0.5 * quadratic)
        )  # (N, K)
        log_r = log_rho - logsumexp(log_rho, axis=1, keepdims=True)  # (N, K)
        # Calculate the responsibility
        r = np.exp(log_r)  # (N, K)
        # Replace the element where nan appears
        r[np.isnan(r)] = 1.0 / (self.K)  # (N, K)
        # Update the optimized r
        self.r = r  # (N, K)

    def m_step(self, X):
        """Execute the variational M-step of VB.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Calculate the optimized N_k
        N_k = np.sum(self.r, 0)  # (K)
        r_tile = np.tile(self.r[:, :, None], (1, 1, self.D)).transpose(
            1, 2, 0
        )  # (K, D, N)
        x_bar = np.sum(
            (r_tile * np.tile(X[None, :, :], (self.K, 1, 1)).transpose(0, 2, 1)), 2
        ) / (N_k[:, None] + np.spacing(1))  # (K, D)
        res_error = np.tile(X[None, :, :], (self.K, 1, 1)).transpose(0, 2, 1) - np.tile(
            x_bar[:, :, None], (1, 1, self.N)
        )  # (K, D, N)
        S = ((r_tile * res_error) @ res_error.transpose(0, 2, 1)) / (
            N_k[:, None, None] + np.spacing(1)
        )  # (K, D, D)
        res_error_bar = x_bar - np.tile(self.m0[None, :], (self.K, 1))  # (K, D)
        # Update the optimized parameters
        self.alpha = self.alpha0 + N_k  # (K)
        self.beta = self.beta0 + N_k  # (K)
        self.nu = self.nu0 + N_k  # (K)
        self.m = (
            np.tile((self.beta0 * self.m0)[None, :], (self.K, 1))
            + (N_k[:, None] * x_bar)
        ) / (self.beta[:, None] + np.spacing(1))  # (K, D)
        W_inv = (
            la.pinv(self.W0)
            + (N_k[:, None, None] * S)
            + (
                (
                    (self.beta0 * N_k)[:, None, None]
                    * res_error_bar[:, :, None]
                    @ res_error_bar[:, None, :]
                )
                / (self.beta0 + N_k)[:, None, None]
                + np.spacing(1)
            )
        )  # (K, D, D)
        self.W = la.pinv(W_inv)  # (K, D, D)

    def plot_highest_density_interval(self, dim, z=1.96):
        """Plot the Highest Density Interval (HDI) for each component.

        Args:
            dim (int): Dimensionality of the data. Use `DIMENSION_1D` for 1D visualization,
                otherwise assumes 2D.
            z (float, optional): Z-score corresponding to the desired confidence level.
                Defaults to 1.96 (approx. 95%).

        """
        for k, (m, W) in enumerate(zip(self.m, self.W, strict=True)):
            sigma = np.diag(W)
            hdi_range = np.array([m - z * np.sqrt(sigma), m + z * np.sqrt(sigma)])
            x = np.linspace(hdi_range[0][0], hdi_range[1][0], 1000)

            label = "HDI" if k == 0 else None
            if dim == DIMENSION_1D:
                plt.fill_between(x, -10, 10, alpha=0.8, label=label)
                plt.ylim(-5, 5)
            else:
                plt.fill_between(
                    x, hdi_range[0][1], hdi_range[1][1], alpha=0.8, label=label
                )

    def visualize(self, X, output_path="./results/clusters.png", HDI=False):
        """Execute the classification.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        _, dim = X.shape

        # Execute classification
        labels = np.argmax(self.r, 1)  # (N)
        # Use the custome color list.
        cm = plt.get_cmap("tab10")
        # Remove ticks
        plt.xticks([])
        plt.yticks([])

        # add 0 when 1D
        if dim == DIMENSION_1D:
            X = np.hstack([X, np.zeros((X.shape[0], 1))])
            centroids = np.hstack([self.m, np.zeros((self.K, 1))])
        else:
            centroids = self.m

        # plot data
        plt.scatter(
            X[:, 0],
            X[:, 1],
            alpha=0.5,
            c=cm(labels),
            label="Data sample",
        )

        # plot centroid
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            color="red",
            label="Centroids",
        )

        # plot HDI
        if HDI:
            self.plot_highest_density_interval(dim)

        # plot probability density
        else:  # noqa: PLR5501
            if dim == DIMENSION_1D:
                x = np.linspace(min(X[:, 0]) * 1.1, max(X[:, 0]) * 1.1, 1000)
                y = np.sum(self.gmm_pdf(x), axis=1)
                plt.plot(x, y, label="Probability density")
            else:
                x = np.linspace(
                    min(X[:, 0]) * 1.1,
                    max(X[:, 0]) * 1.1,
                    200,
                )
                y = np.linspace(
                    min(X[:, 1]) * 1.1,
                    max(X[:, 1]) * 1.1,
                    200,
                )
                x, y = np.meshgrid(x, y)
                z = np.sum(
                    self.gmm_pdf(np.column_stack([x.ravel(), y.ravel()])), axis=1
                ).reshape(x.shape)
                contour = plt.contour(x, y, z)

                # plot continuous colorbar
                norm = matplotlib.colors.Normalize(
                    vmin=contour.cvalues.min(), vmax=contour.cvalues.max()
                )
                sm = plt.cm.ScalarMappable(norm=norm, cmap=contour.cmap)
                sm.set_array([])
                plt.colorbar(sm, ticks=contour.levels, label="Probability density")

        plt.xlabel(r"$x$")
        if dim == DIMENSION_1D:
            plt.ylabel("Probability density")
        else:
            plt.ylabel(r"$y$")
        plt.legend()
        plt.savefig(output_path)
        plt.show()

    def execute(self, X, iter_max, thr):
        """Execute VB.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).
            iter_max (int): Maximum number of updates.
            thr (float): Threshold of convergence condition.

        Returns:
            None.
        """
        # Init the parameters
        self.init_params(X)
        # Prepare the list of the log_likelihood at each iteration
        log_likelihood_list = []
        # Calculate the initial log-likelihood
        log_likelihood_list.append(
            np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
        )
        # Start the iteration
        for i in range(iter_max):
            # Execute E-step
            self.e_step(X)
            # Execute M-step
            self.m_step(X)
            # Add the current log-likelihood
            log_likelihood_list.append(
                np.mean(np.log(np.sum(self.gmm_pdf(X), 1) + self.eps))
            )
            # Print the gap between the previous log-likelihood and current one
            print(
                "Log-likelihood gap: "
                + str(
                    round(
                        np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]), 2
                    )
                )
            )
            # Visualization is performed when the convergence condition is met or when the upper limit is reached
            if (np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]) < thr) or (
                i == iter_max - 1
            ):
                print(f"VB has stopped after {i + 1} iteraions.")
                break


class NameSpace:
    """Configuration namespace for linear regression processing parameters.

    Parameters
    ----------
    input_paths: list[Path]
        List of paths to input data.
    output_dir : Path
        Path where the processed output will be saved.
    cluster_nums : list[int]
        List of cluster numbers to be used in GMMVB.
    iter_max : int
        Maximum number of iterations for the GMMVB.
    threshold : float
        Convergence threshold for the GMMVB.

    """

    input_paths: list[Path]
    output_dir: Path
    cluster_nums: list[int]
    iter_max: int
    threshold: float


def parse_args() -> NameSpace:
    """Parse command-line arguments.

    Returns
    -------
    arguments : NameSpace
        Parsed arguments including input/output paths, cluster numbers parameters.

    """
    # data path
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = (SCRIPT_DIR / "../").resolve()
    DATA1_PATH = DATA_DIR / "data1.csv"
    DATA2_PATH = DATA_DIR / "data2.csv"
    DATA3_PATH = DATA_DIR / "data3.csv"

    # prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output result directory.",
    )
    parser.add_argument(
        "--input_paths",
        type=Path,
        nargs="*",
        default=[DATA1_PATH, DATA2_PATH, DATA3_PATH],
        help="Path to input data.",
    )
    parser.add_argument(
        "--cluster_nums",
        type=int,
        nargs="*",
        default=[2, 3, 3],
        help="The number of cluster for GMMVB.",
    )
    parser.add_argument(
        "--iter_max",
        type=int,
        default=1000,
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Convergence threshold.",
    )

    return parser.parse_args(namespace=NameSpace())


if __name__ == "__main__":
    # get arguments
    args = parse_args()
    input_paths = args.input_paths
    output_dir = args.output_dir
    cluster_nums = args.cluster_nums
    iter_max = args.iter_max
    threshold = args.threshold

    # load data
    data: list[pd.DataFrame] = []
    for input_path in input_paths:
        df = pd.read_csv(input_path, header=None)
        df.attrs["title"] = input_path.stem
        data.append(df)

    # plot GMMVB results
    np.random.seed(42)
    for df, cluster_num in zip(data, cluster_nums, strict=True):
        title = df.attrs["title"]
        model = GMMVB(cluster_num)
        model.execute(df.to_numpy(), iter_max=iter_max, thr=threshold)
        model.visualize(df.to_numpy(), output_path=output_dir / f"{title}_cluster.png")
        model.visualize(
            df.to_numpy(), output_path=output_dir / f"{title}_HDI.png", HDI=True
        )
