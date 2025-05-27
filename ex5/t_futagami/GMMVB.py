"""
GMMVBクラスを実装するモジュール.

パラメータを推定し, HDIを可視化する.
"""

from collections import Counter

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy import linalg as la
from scipy.special import digamma, logsumexp
from scipy.stats import chi2, multivariate_normal


class GMMVB:
    """
    ガウス混合モデル(GMM)の変分ベイズ推定を実行するクラス.

    Parameters:
        K (int): クラスタ数.
    Attributes:
        eps (float): オーバーフローとアンダーフローを防ぐための小さな値.
        N (int): データのサンプル数.
        D (int): データの次元数.
        alpha0 (float): 初期の重み係数.
        beta0 (float): 初期のスケールパラメータ.
        nu0 (float): 初期の自由度パラメータ.
        m0 (numpy ndarray): 初期の平均ベクトル.
        W0 (numpy ndarray): 初期の共分散行列.
        r (numpy ndarray): 責任行列.
        alpha (numpy ndarray): 更新された重み係数.
        beta (numpy ndarray): 更新されたスケールパラメータ.
        nu (numpy ndarray): 更新された自由度パラメータ.
        m (numpy ndarray): 更新された平均ベクトル.
        W (numpy ndarray): 更新された共分散行列.
    """

    def __init__(self, K):
        """Initialize the GMMVB class.

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
            Probability density function (numpy ndarray):
                The probability density function of the GMM at each data point.
                Size is (N, K).
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
        ) + (
            self.D * np.log(2) + (np.log(la.det(self.W) + np.spacing(1)))
        )  # (K)
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
        ) / (
            N_k[:, None] + np.spacing(1)
        )  # (K, D)
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
        ) / (
            self.beta[:, None] + np.spacing(1)
        )  # (K, D)
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

    def visualize(self, X):
        """Visualize the classification results.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        if self.D == 1:
            self.visualize_1d(X)
        elif self.D == 2:
            self.visualize_2d(X)
        else:
            print("Visualization is not supported for dimensions other than 1 and 2.")

    def visualize_1d(self, X):
        """Execute the classification for 1D data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Execute classification
        labels = np.argmax(self.r, 1)  # (N)
        # Visualize each clusters
        label_frequency_desc = [label[0] for label in Counter(labels).most_common()]
        # Prepare the visualization
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        # Use the custome color list.
        cm = plt.get_cmap("tab10")
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        for order, label in enumerate(label_frequency_desc):
            idx = np.where(labels == label)[0]
            cols = np.tile(cm(order), (len(idx), 1))
            cols[:, 3] = self.r[idx, label]
            ax.scatter(
                X[idx, 0],
                np.zeros_like(X[idx, 0]),
                color=cols,
                s=1,
                marker="o",
            )
            z = 1.96  # 95% confidence interval
            cov = la.pinv(self.nu[label] * self.W[label])
            sigma = np.sqrt(cov[0, 0])
            hdi = (
                self.m[label, 0] - z * sigma,
                self.m[label, 0] + z * sigma,
            )
            height = 2
            y_min = -height / 2
            legend_label = "HDI" if order == 0 else None
            rect = plt.Rectangle(
                (hdi[0], y_min),
                hdi[1] - hdi[0],
                height,
                color=cm(order),
                alpha=0.5,
                label=legend_label,
            )
            ax.add_patch(rect)

        # Plot the means of each cluster
        means = self.m[label_frequency_desc, :]  # (K, D)
        ax.scatter(
            means[:, 0],
            np.zeros_like(means[:, 0]),
            color="black",
            marker="x",
            s=100,
            label="means",
        )

        ax.set_xlabel("x")
        ax.set_title("data")
        ax.set_ylim(-1, 1)
        ax.legend()
        fig.savefig("./figs/clusters.png")

    def visualize_2d(self, X):
        """Execute the classification for 2D data.

        Args:
            X (numpy ndarray): Input data whose size is (N, D).

        Returns:
            None.
        """
        # Execute classification
        labels = np.argmax(self.r, 1)  # (N)
        # Visualize each clusters
        label_frequency_desc = [label[0] for label in Counter(labels).most_common()]
        # Prepare the visualization
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        # Use the custome color list.
        cm = plt.get_cmap("tab10")
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        for order, label in enumerate(label_frequency_desc):
            idx = np.where(labels == label)[0]
            cols = np.tile(cm(order), (len(idx), 1))
            cols[:, 3] = self.r[idx, label]
            ax.scatter(
                X[idx, 0],
                X[idx, 1],
                color=cols,
                s=1,
                marker="o",
            )
        # Plot the means of each cluster
        means = self.m[label_frequency_desc, :]  # (K, D)
        ax.scatter(
            means[:, 0], means[:, 1], color="black", marker="x", s=100, label="means"
        )
        # HDI ellipses (95% interval)
        chi2_val = chi2.ppf(0.95, df=2)
        for order, label in enumerate(label_frequency_desc):
            # approximate covariance from posterior precision
            cov = la.pinv(self.nu[label] * self.W[label])
            # eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx_sort = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[idx_sort], eigvecs[:, idx_sort]
            # ellipse params
            width, height = 2 * np.sqrt(eigvals * chi2_val)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            legend = "HDI" if order == 0 else None
            ellipse = Ellipse(
                self.m[label],
                width,
                height,
                angle=angle,
                edgecolor=cm(order),
                facecolor="none",
                lw=1.5,
                label=legend,
            )
            ax.add_patch(ellipse)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        ax.set_title("data")
        fig.savefig("./figs/clusters.png")

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
            # Check the convergence condition
            if (np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]) < thr) or (
                i == iter_max - 1
            ):
                print(f"VB has stopped after {i + 1} iteraions.")
                self.visualize(X)
                break


def main(K):
    """Execute the GMMVB algorithm.

    Args:
        K (int): The number of clusters.
    Returns:
        GMMVB instance: An instance of the GMMVB class initialized with K clusters.
    """
    return GMMVB(K)


if __name__ == "__main__":
    fire.Fire(main)
