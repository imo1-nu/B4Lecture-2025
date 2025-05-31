"""
HMMのForwardアルゴリズムとViterbiアルゴリズムを実装するプログラム.

ForwardアルゴリズムとViterbiアルゴリズムを用いて、HMMの出力データに対する予測を行い、結果を可視化する.
"""

import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np


class HMM:
    """
    HMMのForwardアルゴリズムとViterbiアルゴリズムを実装するクラス.

    Attributes:
        answer_models (list): 正解のモデルのリスト. shape (P,).
        outputs (list): 出力データのリスト. shape (P, T).
        models (dict): HMMのパラメータを含む辞書.
            - "PI": 初期状態確率. shape (K, L, 1).
            - "A": 状態遷移確率行列. shape (K, L, L).
            - "B": 観測確率行列. shape (K, L, N).
    """

    def __init__(self, data):
        """
        HMMクラスの初期化.

        Parameters:
            data (dict): HMMのパラメータを含む辞書.
                - "answer_models": 正解のモデルのリスト.
                - "output": 出力データのリスト.
                - "models": HMMのパラメータを含む辞書.
        """
        self.answer_models = data["answer_models"]
        self.outputs = data["output"]
        self.models = data["models"]

    def forward(
        self,
        output: np.ndarray,
        start_prob: np.ndarray,
        transition_prob: np.ndarray,
        emission_prob: np.ndarray,
    ) -> float:
        """
        Forwardアルゴリズムを実行するメソッド.

        Parameters:
            output (np.ndarray): 出力データ.
            start_prob (np.ndarray): 初期状態確率.
            transition_prob (np.ndarray): 状態遷移確率行列.
            emission_prob (np.ndarray): 観測確率行列.
        Returns:
            float: 出力データの確率.
        """
        T = len(output)  # 出力の長さ
        alpha = start_prob.squeeze() * emission_prob[:, output[0]]  # 初期化
        # αの計算
        for t in range(1, T):
            alpha = alpha @ transition_prob * emission_prob[:, output[t]]
        return np.sum(alpha)  # 出力データの確率を返す

    def viterbi(
        self,
        output: np.ndarray,
        start_prob: np.ndarray,
        transition_prob: np.ndarray,
        emission_prob: np.ndarray,
    ) -> float:
        """
        Viterbiアルゴリズムを実行するメソッド.

        Parameters:
            output (np.ndarray): 出力データ.
            start_prob (np.ndarray): 初期状態確率.
            transition_prob (np.ndarray): 状態遷移確率行列.
            emission_prob (np.ndarray): 観測確率行列.
        Returns:
            float: 最も可能性の高い状態遷移の確率.
        """
        T = len(output)  # 出力の長さ
        delta = start_prob.squeeze() * emission_prob[:, output[0]]  # 初期化
        # δの計算
        for t in range(1, T):
            delta = (
                np.max(delta.reshape(-1, 1) * transition_prob, axis=0)
                * emission_prob[:, output[t]]
            )
        return np.max(delta)  # 最も可能性の高い状態遷移の確率を返す

    def visualize(self, predict_models: np.ndarray, title: str, file_name: str) -> None:
        """
        予測結果を可視化するメソッド.

        Parameters:
            predict_models (np.ndarray): 予測されたモデルの配列.
            title (str): グラフのタイトル.
            file_name (str): 保存するファイル名.
        """
        K = len(self.models["PI"])  # モデルの数
        table = np.zeros((K, K))  # 混同行列の初期化
        # 混同行列の作成
        for answer_model, predict_model in zip(self.answer_models, predict_models):
            table[answer_model, predict_model] += 1
        accuracy = np.diag(table).sum() / table.sum() * 100  # 精度の計算

        # 混同行列の可視化
        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
        ax.matshow(table, cmap="Blues")
        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        for (i, j), val in np.ndenumerate(table):
            text_color = "white" if val > table.max() / 2 else "black"
            ax.text(
                j, i, int(val), ha="center", va="center", color=text_color, fontsize=12
            )
        ax.set_xticklabels([i + 1 for i in range(K)], fontsize=12)
        ax.set_yticklabels([i + 1 for i in range(K)], fontsize=12)
        ax.set_xlabel("Predicted model", fontsize=12)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Actual model", fontsize=12)
        ax.set_title(f"{title}\nAcc: {accuracy:.0f}%", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"./figs/{file_name}.png")

    def execute(self) -> None:
        """
        ForwardアルゴリズムとViterbiアルゴリズムを実行し、結果を可視化するメソッド.

        各アルゴリズムの実行時間を計測し、平均時間を表示する。
        """
        K = len(self.models["PI"])  # モデルの数
        predict_forward = []  # Forwardアルゴリズムの予測結果
        predict_viterbi = []  # Viterbiアルゴリズムの予測結果
        time_forward = []  # Forwardアルゴリズムの実行時間
        time_viterbi = []  # Viterbiアルゴリズムの実行時間

        # 各出力データに対してForwardアルゴリズムとViterbiアルゴリズムを実行
        for output in self.outputs:
            # Forwardアルゴリズムの実行
            start = time.perf_counter()
            forward_results = [
                self.forward(
                    output,
                    self.models["PI"][k],
                    self.models["A"][k],
                    self.models["B"][k],
                )
                for k in range(K)
            ]
            time_forward.append(time.perf_counter() - start)

            # Viterbiアルゴリズムの実行
            start = time.perf_counter()
            viterbi_results = [
                self.viterbi(
                    output,
                    self.models["PI"][k],
                    self.models["A"][k],
                    self.models["B"][k],
                )
                for k in range(K)
            ]
            time_viterbi.append(time.perf_counter() - start)

            # 予測結果の保存
            predict_forward.append(np.argmax(forward_results))
            predict_viterbi.append(np.argmax(viterbi_results))

        # 平均実行時間の計算と結果の表示
        print(f"Forward algorithm time: {np.sum(time_forward):.4f} seconds")
        print(f"Viterbi algorithm time: {np.sum(time_viterbi):.4f} seconds")
        self.visualize(
            np.array(predict_forward),
            title="Forward algorithm",
            file_name="result_forward",
        )
        self.visualize(
            np.array(predict_viterbi),
            title="Viterbi algorithm",
            file_name="result_viterbi",
        )


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.

    入力データのパスを取得.
    """
    parser = argparse.ArgumentParser(description="HMMをpickleファイルから読み込む.")
    parser.add_argument(
        "input_data",
        type=str,
        help="入力データのパスを指定してください.",
    )

    args = parser.parse_args()

    # 入力の検証
    if not args.input_data.endswith(".pickle"):
        parser.error("入力データはpickleファイルである必要があります。")

    return args


def main():
    """
    メイン関数.

    コマンドライン引数を解析し、HMMクラスを初期化して実行する。
    """
    args = parse_arguments()
    data = pickle.load(open(args.input_data, "rb"))  # データの読み込み
    hmm = HMM(data)  # HMMクラスの初期化
    hmm.execute()  # HMMの実行


if __name__ == "__main__":
    main()
