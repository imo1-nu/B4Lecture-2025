"""線形回帰・主成分分析の実装.

線形回帰・主成分分析を行うプログラムを実装する.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_CLASTER = 3  # クラスター数

def read_csv(path: str) -> np.ndarray:
    """csvファイルを読み込む関数.

    指定されたパスからcsvファイルを読み込み，numpy配列として返す.
    入力：
      path(str): csvファイルのパス
    出力：
      csv_data(np.ndarray).shape = (行番号, 行データ): csvデータのnumpy配列
    """
    if not os.path.exists(path):
        return np.array([])

    data = csv.reader(open(path, "r"))
    data = [list(map(float, row)) for row in data]  # 各要素をfloatに変換
    data = np.array(data).T

    return data


def make_scatter(
    data: np.ndarray, title: str, claster: np.ndarray = None, mean: np.ndarray = None
) -> None:
    """散布図を作成する関数.

    入力：
        data(np.ndarray): 散布図のデータ, shape = (2, data) || (3, data)
        title(str): グラフのタイトル
    出力：
        None(散布図を出力)
    """
    fig = plt.figure()

    if claster is None:
        # 散布図を描画する場合の分岐
        if data.shape[0] == 2:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[0], data[1])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        elif data.shape[0] == 1:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[0], np.zeros(np.shape(data[0])))
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    else:
        # クラスタリング結果を描画する場合の分岐
        if data.shape[0] == 1:
            ax = fig.add_subplot(1, 1, 1)
            for i in range(NUMBER_OF_CLASTER):
                ax.scatter(data[0][claster == i], 0, label=f"Claster {i}")
            ax.scatter(mean[:, 0], 0, c="red", marker="x", s=100, label="Mean")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
        elif data.shape[0] == 2:
            ax = fig.add_subplot(1, 1, 1)
            for i in range(NUMBER_OF_CLASTER):
                ax.scatter(
                    data[0][claster == i], data[1][claster == i], label=f"Claster {i}", cmap="coolwarm"
                )
            plt.scatter(
                mean[:, 0], mean[:, 1], c="red", marker="x", s=100, label="means"
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()

    plt.show()

def plot_gmm_contour(data, gmm):
    from scipy.stats import multivariate_normal
    """GMMの混合分布等高線を描画（2次元データのみ）"""
    if data.shape[0] != 2:
        print("等高線は2次元データのみ対応しています")
        return

    x = np.linspace(np.min(data[0]), np.max(data[0]), 100)
    y = np.linspace(np.min(data[1]), np.max(data[1]), 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()]).T

    # 混合分布のPDFを計算
    pdf = np.zeros(grid.shape[0])
    for k in range(gmm.claster):
        rv = multivariate_normal(mean=gmm.mean[k], cov=gmm.cov[k])
        pdf += gmm.mixture_ratio[k] * rv.pdf(grid)
    pdf = pdf.reshape(xx.shape)

    plt.figure()
    plt.scatter(data[0], data[1], c='gray', s=10, label='data')
    plt.contour(xx, yy, pdf, levels=10, cmap='viridis')
    plt.scatter(gmm.mean[:, 0], gmm.mean[:, 1], c='red', marker='x', s=100, label='means')
    plt.title("GMM Contour")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


class GMMClastering:
    """GMMを利用したクラスタリングの実装を行うクラス.

    メンバ関数：
        _GaussianModel: ガウス分布の確率密度関数を計算する内生関数
        _LogLikelihood: 対数尤度を計算する内生関数
        EM_Algorithm: EMアルゴリズムを実装する関数
        _E_Step: EMアルゴリズムのEステップを実装する内生関数
        _M_Step: EMアルゴリズムのMステップを実装する内生関数
        clastering: クラスタリングを行う関数
        define_claster: AICまたはBICを用いてクラスター数を決定する関数

    属性：
        data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
        claster(int): クラスター数
        mixture_ratio(float): 混合比率
        mean(np.ndarray): 平均, shape = (次数)
        cov(np.ndarray): 共分散行列, shape = (次数, 次数) 
    """
    def __init__(self, data: np.ndarray) -> None:
        """初期化関数.

        クラスタリングの初期化を行う関数. 変数の受領及び初期化を行う.

        入力：
            M(int): 次数
            path(str): データファイルのパス
            model(str): モデルの種類
        """
        self.data = data
        self.n_samples = data.shape[1]
        self.n_dimensions = data.shape[0]
        self.claster = NUMBER_OF_CLASTER

        self.mixture_ratio = np.ones(self.claster) / self.claster
        self.mean = self.data[:, np.random.choice(self.n_samples, self.claster, replace=False)].T # shape = (クラスター数, 次数)
        self.cov = np.array([np.cov(data) for _ in range(self.claster)])

        self.responsibility = np.zeros((self.claster, self.n_samples))  # 責任度
        self._LogLikelihoods = [] # 対数尤度のリスト, shape = (イテレーション数, )

    def _GaussianModel(self, mean, cov):
        """ガウス分布の確率密度関数を計算する関数.

        入力：
            mean(np.ndarray): 平均, shape = (次数)
            cov(np.ndarray): 共分散行列, shape = (次数, 次数)
        出力：
            likelihood(np.ndarray): ガウス分布の確率密度関数, shape = (クラスター数, サンプル数)
        """
        inv_cov = np.linalg.pinv(cov)  # shape = (次元数, 次元数)
        det_cov = np.linalg.det(cov)
        
        diff = self.data.T - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)  # shape = (サンプル数,)
        normalization = np.sqrt((2 * np.pi) ** self.n_dimensions * det_cov)
        return (1 / normalization) * np.exp(exponent)  # shape = (サンプル数,)

    def _LogLikelihood(self):
        """対数尤度を計算する関数.

        入力：
            mixture_ratio(float): 混合比率
            mean(np.ndarray): 平均, shape = (クラスター数, 次数)
            cov(np.ndarray): 共分散行列, shape = (クラスター数, 次数, 次数)
        出力：
            L(float): 対数尤度
        """
        # ガウス分布の確率密度関数を計算
        likelihood = np.zeros((self.claster, self.data.shape[1]))  # shape = (クラスター数, サンプル数)
        for k in range(self.claster):
            likelihood[k] = self.mixture_ratio[k] * self._GaussianModel(self.mean[k], self.cov[k])  # shape = (クラスター数, サンプル数)
        total_likelihood = np.sum(likelihood, axis=0)  # shape = (サンプル数,)
        # 対数尤度を計算
        L = np.sum(np.log(total_likelihood + 1e-10))
        return L

    def EM_Algorithm(self):
        """EMアルゴリズムの実装を行う関数.

        入力：
            data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
            claster(int): クラスター数

        出力：
            responsibility(np.ndarray): 責任度, shape = (データ数, クラスター数)
        """
        L = 0
        L_new = self._LogLikelihood()
        self._LogLikelihoods.append(L_new)
        responsibility = np.zeros((self.claster, self.n_samples))  # 責任度
        for i in range(100):
            L = L_new
            responsibility = self._E_Step()
            self._M_Step(responsibility)
            L_new = self._LogLikelihood()
            self._LogLikelihoods.append(L_new)
            if abs(L_new - L) < 1e-3:
                break

        self.responsibility = responsibility
        self.display_log_likelihood()

    def _E_Step(self):
        """EMアルゴリズムのEステップを実装する関数.

        入力：
            mixture_ratio(float): 混合比率
            mean(np.ndarray): 平均, shape = (次数)
            cov(np.ndarray): 共分散行列, shape = (次数, 次数)
        出力：
            yupsilon(np.ndarray): 責任度, shape = (データ数, クラスター数)
        """
        # ガウス分布の確率密度関数を計算
        gaussians = np.zeros((self.claster, self.data.shape[1]))  # shape = (クラスター数, サンプル数)
        for k in range(self.claster):
            gaussians[k] = self._GaussianModel(self.mean[k], self.cov[k])  # shape = (クラスター数, サンプル数)
        # 責任度 (yupsilon) を計算
        yupsilon = np.zeros((self.claster, self.n_samples))  # shape = (クラスター数, サンプル数)
        for k in range(self.claster):
            yupsilon[k] = self.mixture_ratio[k] * gaussians[k] / np.sum(self.mixture_ratio[:, np.newaxis] * gaussians, axis=0)  # shape = (クラスター数, サンプル数)
        return yupsilon

    def _M_Step(self, responsibility) -> None:
        """EMアルゴリズムのMステップを実装する関数.

        入力：
            responsibility(np.ndarray): 責任度, shape = (クラスター数, サンプル数)
        出力：
            mixture_ratio(np.ndarray): 混合比率, shape = (クラスター数)
            mean(np.ndarray): 平均, shape = (クラスター数, 次数)
            cov(np.ndarray): 共分散行列, shape = (クラスター数, 次数, 次数)
        """
        Nk = np.sum(responsibility, axis=1)
        for k in range(self.claster):
            self.mean[k] = np.sum(self.data * responsibility[k], axis=1) / Nk[k]

            diff = self.data - self.mean[k][:, np.newaxis]
            cov_k = (responsibility[k] * diff) @ diff.T / Nk[k]
            self.cov[k] = cov_k + np.eye(self.n_dimensions) * 1e-6

            self.mixture_ratio[k] = Nk[k] / self.n_samples
    def clustering(self) -> np.ndarray:
        """クラスタリングを行う関数.

        入力：
            responsibility(np.ndarray): 責任度, shape = (クラスター数, サンプル数)
        出力：
            claster(np.ndarray): クラスタリング結果, shape = (サンプル数)
        """
        # クラスタリング結果を取得
        self.EM_Algorithm()
        claster = np.argmax(self.responsibility, axis=0)
        return claster
    
    def define_claster(self, data: np.ndarray) -> int:
        return 3

    def display_log_likelihood(self) -> None:
        """対数尤度を表示する関数.

        入力：
            _logLikelihoods(List<float>): 対数尤度のリスト
        出力：
            None
        """
        print(self._LogLikelihoods)
        plt.plot(self._LogLikelihoods)
        plt.title("Log Likelihood")
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.show()

def parse_arguments():
    """コマンドライン引数を解析する関数.

    入力：
        なし

    出力：
        args: 入力のオブジェクト
    """
    import argparse

    parser = argparse.ArgumentParser(description="GMMを利用したクラスタリング")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="データのパス",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """メイン関数."""

    args = parse_arguments()

    # データの読み込み
    data = read_csv(args.path)
    make_scatter(data, f"Scatter of {args.path.split('/')[-1].split('.')[0]}")

    gmm = GMMClastering(data)
    claster = gmm.clustering()
    make_scatter(data, f"Clastering of {args.path.split('/')[-1].split('.')[0]}", claster, gmm.mean)

    # GMMの混合分布等高線を描画
    plot_gmm_contour(data, gmm)