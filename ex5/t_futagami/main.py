import argparse

import numpy as np
from GMMVB import GMMVB  # GMMVBクラスをインポート

# English comments for the functions


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.

    入力データのパスを取得.
    """
    parser = argparse.ArgumentParser(description="GMMベースのクラスタリングの実行")
    parser.add_argument(
        "input_data",
        type=str,
        help="入力データのパスを指定してください。",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="GMMのコンポーネント数を指定します。デフォルトは3です。",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="GMMの最大反復回数を指定します。デフォルトは100です。",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="GMMの収束許容誤差を指定します。デフォルトは1e-3です。",
    )

    args = parser.parse_args()
    # 入力の検証
    if not args.input_data.endswith(".csv"):
        parser.error("入力データはCSVファイルである必要があります。")

    return args


def main():
    """
    メイン関数.

    コマンドライン引数を解析し、GMMを学習させ、
    対数尤度の変化とクラスタリング結果をプロットする.
    """
    args = parse_arguments()
    X = np.genfromtxt(args.input_data, delimiter=",")  # CSVファイルからデータを読み込む
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    gmm = GMMVB(args.n_components)  # GMMVBクラスのインスタンスを作成
    gmm.execute(X, iter_max=args.max_iter, thr=args.tol)  # GMMの学習を実行


if __name__ == "__main__":
    main()
