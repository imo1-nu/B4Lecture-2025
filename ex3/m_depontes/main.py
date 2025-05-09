"""線形回帰・主成分分析の実装.

線形回帰・主成分分析を行うプログラムを実装する．
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import csv


def read_csv(path: str) -> np.ndarray:
    """csvファイルを読み込む関数.

    指定されたパスからcsvファイルを読み込み，numpy配列として返す．
    入力：
      path(str): csvファイルのパス
    出力：
      csv_data(np.ndarray).shape = (行番号, 行データ): csvデータのnumpy配列
    """
    if not os.path.exists(path):
        return np.array([])
    
    data = csv.reader(open(path, "r"))
    data = [row for row in data][1:]  # ヘッダーを除外
    data = [list(map(float, row)) for row in data]  # 各要素をfloatに変換
    data = np.array(data).T
    
    return data

def make_scatter(data: np.ndarray, title: str, pred: np.ndarray = None) -> None:
    """散布図を作成する関数.

    入力：
        data(np.ndarray): 散布図のデータ, shape = (2, data) || (3, data)
        title(str): グラフのタイトル
    出力：
        None(散布図を出力)
    """
    fig = plt.figure()
    if pred is None:
        if data.shape[0] == 2:
            
            ax = fig.add_subplot(1,1,1)
            ax.scatter(data[0], data[1])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif data.shape[0] == 3:
            ax = fig.add_subplot(projection="3d")
            ax.scatter(data[0], data[1], data[2])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

    else:
        if data.shape[0] == 2:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[0], data[1])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            sorted_indices = np.argsort(data[0])
            plt.plot(data[0][sorted_indices], pred[sorted_indices], color='red')
            plt.legend(["Actual", "Predicted"])
        elif data.shape[0] == 3:
            ax = fig.add_subplot(projection="3d")
            ax.scatter(data[0], data[1], data[2])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            # 回帰平面の描画
            x = np.linspace(min(data[0]), max(data[0]), 50)
            y = np.linspace(min(data[1]), max(data[1]), 50)
            X, Y = np.meshgrid(x, y)
            Z = pred[0] + pred[1] * X + pred[2] * Y  # 平面の方程式
            ax.plot_surface(X, Y, Z, color='red', alpha=0.5)
    plt.show()


class LinearRegression:
    """線形回帰の実装を行うクラス.

    メンバ関数：
        OLS: 最小二乗法の計算を行う関数
        fit: データへの適応を行う関数

    属性：
        M(int): 次数
        fc(int): カットオフ周波数．複数の場合は下限を意味する．
        fs(int): サンプリング周波数
        fc2(int): 上限カットオフ周波数
    """

    def __init__(self,M: int, path: str, model: str) -> None:
        """初期化関数.

        入力：
            M: 次数
        """
        self.M = M
        self.w = None  # 重み
        self.path = path
        self.data = np.array([])
        self.model = model

    def OLS(self, M, data):
        """畳み込み計算を行う関数.

        入力：
            data(np.ndarray): 入力値, shape = (入力値,入力値に対する出力値)
            M(int): 次元数

        出力：
            w: 重み
        """
        x = data[0]  # 入力値
        y = data[1]  # 出力値

        # 入力行列 X の構築 (バイアス項を含む)
        X = np.vstack([x**i for i in range(M)]).T  # shape = (サンプル数, M)
        # 正規方程式を用いて重みを計算
        match self.model:
            case "normal":
                w = np.linalg.inv(X.T @ X) @ X.T @ y
                return w
            case "Ridge":
                alpha = 0.1  # 正則化パラメータ
                w = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
                return w
            case "Lasso":
                alpha = 0.1  # 正則化パラメータ
                w = self.elastic_net_coordinate_descent(X, y, alpha)
                return w
            case "Elastic":
                alpha = 0.1  # 正則化パラメータ
                l1_ratio = 0.5  # L1正則化の比率（0～1）
                w = self.elastic_net_coordinate_descent(X, y, alpha, l1_ratio)
                return w
        
    def elastic_net_coordinate_descent(self, X, y, alpha, l1_ratio=1.0, max_iter=1000, tol=1e-4):
        """Elastic Net回帰及びLasso回帰を座標降下法で解く関数.

        入力：
            X(np.ndarray): 入力行列, shape = (サンプル数, 特徴量数)
            y(np.ndarray): 出力値, shape = (サンプル数,)
            alpha(float): 正則化パラメータ
            l1_ratio(float): L1正則化の比率（0～1, デフォルトは1.0でLasso回帰）
            max_iter(int): 最大反復回数
            tol(float): 収束判定の閾値

        出力：
            w(np.ndarray): 重みベクトル, shape = (特徴量数,)
        """
        m, n = X.shape
        w = np.zeros(n)  # 初期重みをゼロで初期化

        for iteration in range(max_iter):
            w_old = w.copy()
            for j in range(n):
                # 残差を計算
                residual = y - (X @ w - X[:, j] * w[j])
                # 重みの更新
                rho = X[:, j].T @ residual
                l1_term = alpha * l1_ratio
                l2_term = alpha * (1 - l1_ratio)
                if rho < -l1_term:
                    w[j] = (rho + l1_term) / (X[:, j].T @ X[:, j] + l2_term)
                elif rho > l1_term:
                    w[j] = (rho - l1_term) / (X[:, j].T @ X[:, j] + l2_term)
                else:
                    w[j] = 0

            # 収束判定
            if np.linalg.norm(w - w_old, ord=2) < tol:
                break

        return w


    def fit(self) -> None:
        """データを読み込み、線形回帰を行う関数.

        入力：
            M(int): フィルタの次数
            fc(int): 下側のカットオフ周波数
            fs(int): サンプリング周波数

        出力：
            filter(np.ndarray): フィルタ係数, shape = (filterValue)
        """

        self.data = read_csv(self.path)

        self.w = self.OLS(self.M, self.data)

        

    def predict(self) -> np.ndarray:
        x = self.data[0]
        X = np.vstack([x**i for i in range(self.w.shape[0])]).T
        y_pred = (X @ self.w).flatten()  # 予測値
        
        return self.data,y_pred
        

def parse_arguments():
    """コマンドライン引数を解析する関数.

    入力：
        なし

    出力：
        args: 入力のオブジェクト
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="線形回帰・主成分分析の実装"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["linear","pca"],
        help="実行する処理のモードを指定 (linear: 線形回帰, pca: 主成分分析)",
    )
    parser.add_argument(
        "--path",
        type=float,
        nargs="*",
        required=True,
        help="データのパス",
    )
    parser.add_argument(
        "--M",
        type=int,
        required=True,
        help="次数",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="normal",
        choises=["normal","Ridge","Lasso","Elastic"],
        help="(線形回帰の場合)モデルを指定（標準：normal）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """メイン関数.
    """

    args = parse_arguments()

    linearRegression = LinearRegression(path=args.path,M=args.M,model=args.model)
    linearRegression.fit(args.path)
    data,pred = linearRegression.predict()
    make_scatter(data, "linearRegression", pred)


    # # FIRフィルタのインスタンスを生成
    # if len(args.fc) == 1:
    #     fir = FIR(M=args.M, fc=args.fc[0], fs=args.fs)
    # else:
    #     fir = FIR(M=args.M, fc=args.fc[0], fc2=args.fc[1], fs=args.fs)

    # # 入力信号の読み込み
    # audio = read_audio(args.input)
    # if audio.size == 0:
    #     raise FileNotFoundError(f"指定されたファイルが見つかりません: {args.input}")

    # # フィルタの種類に応じて処理を分岐
    # if args.filter == "low":
    #     filter_coeff = fir.low_path_filter()
    #     title = "Low-Pass Filter"
    # elif args.filter == "high":
    #     filter_coeff = fir.high_pass_filter()
    #     title = "High-Pass Filter"
    # elif args.filter == "bandpass":
    #     if len(args.fc) != 2:
    #         raise ValueError(
    #             "bandpassフィルタには2つのカットオフ周波数を指定してください"
    #         )
    #     filter_coeff = fir.band_pass_filter()
    #     title = "Band-Pass Filter"
    # elif args.filter == "bandstop":
    #     if len(args.fc) != 2:
    #         raise ValueError(
    #             "bandstopフィルタには2つのカットオフ周波数を指定してください"
    #         )
    #     filter_coeff = fir.band_stop_filter()
    #     title = "Band-Stop Filter"

