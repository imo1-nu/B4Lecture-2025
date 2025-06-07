#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト
特徴量；MFCCの平均（0次項含まず）
識別器；MLP
"""

from __future__ import division, print_function

import argparse

import keras_tuner as kt
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def my_MLP(input_shape, output_dim, units1=256, units2=256, dropout1=0.2, dropout2=0.2):
    """
    MLPモデルの構築
    Args:
        input_shape: 入力の形
        output_dim: 出力次元
    Returns:
        model: 定義済みモデル
    """

    model = Sequential()

    model.add(Dense(units1, input_dim=input_shape))
    model.add(Activation("relu"))
    model.add(Dropout(dropout1))

    model.add(Dense(units2))
    model.add(Activation("relu"))
    model.add(Dropout(dropout2))

    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    # モデル構成の表示
    model.summary()

    return model


def extract_segments_mfcc(data, n_mfcc=100, n_segments=4):
    """
    音声データからMFCC特徴量を抽出し、指定された数のセグメントに分割する
    Args:
        data: 音声データ
        n_mfcc: MFCCの次元数
        n_segments: 分割するセグメント数
    Returns:
        segments: 分割されたMFCC特徴量のリスト
    """
    features = []
    for d in data:
        segments = np.array_split(d, n_segments)
        for segment in segments:
            segment *= np.hamming(len(segment))
        mfcc_segments = np.array(
            [
                np.mean(librosa.feature.mfcc(y=segment, n_mfcc=n_mfcc), axis=1)
                for segment in segments
            ]
        )
        # [n_mfcc * n_segments]の形に変形
        feature = mfcc_segments.flatten()
        features.append(feature)
    return np.array(features)


def feature_extraction(train_path_list, test_path_list, cumulative_variance_ratio=0.95):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        train_path_list: 学習データのwavファイルリスト
        test_path_list: テストデータのwavファイルリスト
        cumulative_variance_ratio: 累積寄与率の閾値（デフォルトは0.9）
    Returns:
        features: 特徴量
    """

    load_data = lambda path: librosa.load(path)[0]

    train_data = list(map(load_data, train_path_list))
    test_data = list(map(load_data, test_path_list))

    train_features = extract_segments_mfcc(train_data)
    test_features = extract_segments_mfcc(test_data)

    # PCAを適用して次元削減
    pca = PCA()
    pca.fit(train_features)
    contribution_ratios = pca.explained_variance_ratio_
    cumulative_variance_ratio_ = np.cumsum(contribution_ratios)
    n_components = np.argmax(cumulative_variance_ratio_ >= cumulative_variance_ratio) + 1
    pca_final = PCA(n_components=n_components)
    train_features = pca_final.fit_transform(train_features)
    test_features = pca_final.transform(test_features)

    return train_features, test_features


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """

    cm = confusion_matrix(predict, ground_truth)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig("result/cm.png", transparent=True)
    plt.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """

    with open("result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def plot_history(history):
    # 学習過程をグラフで出力
    # print(history.history.keys())
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc)
    plt.grid()
    plt.title("Model accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("result/history_acc.png", transparent=True)
    plt.show()

    plt.figure()
    plt.plot(epochs, loss)
    plt.grid()
    plt.title("Model loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("result/history_loss.png", transparent=True)
    plt.show()


def build_model_for_tuning(hp, input_shape_val, output_dim_val):
    """
    Keras Tunerがハイパーパラメータを探索するために呼び出すモデル構築関数
    Args:
        hp: Keras TunerのHyperParametersオブジェクト
        input_shape_val: 入力データの形状（特徴量の次元数）
        output_dim_val: 出力層の次元数（クラス数）
    Returns:
        コンパイル済みのKerasモデル
    """
    model = Sequential()

    # 1層目のDense層のユニット数を探索
    hp_units_1 = hp.Int("units1", min_value=64, max_value=512, step=64)
    model.add(Dense(units=hp_units_1, activation="relu", input_dim=input_shape_val))
    # 1層目のDropout率を探索
    hp_dropout_1 = hp.Float("dropout1", min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_1))

    # 2層目のDense層のユニット数を探索
    hp_units_2 = hp.Int("units2", min_value=64, max_value=512, step=64)
    model.add(Dense(units=hp_units_2, activation="relu"))
    # 2層目のDropout率を探索
    hp_dropout_2 = hp.Float("dropout2", min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_2))

    model.add(Dense(output_dim_val, activation="softmax"))

    # 学習率を探索
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス")
    args = parser.parse_args()
    path_to_e7 = "../"

    # データの読み込み
    training = pd.read_csv("../training.csv")
    test = pd.read_csv("../test.csv")

    # 学習データの特徴抽出
    X, X_test = feature_extraction(
        path_to_e7 + training["path"].values, path_to_e7 + test["path"].values
    )

    # 特徴量の次元数を表示
    print(f"Number of features: {X.shape[1]}")

    # 正解ラベルをone-hotベクトルに変換 ex. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    Y = to_categorical(y=training["label"], num_classes=10)

    # ハイパーパラメータチューニング用のデータ分割 (学習データの一部を使用)
    # test_sizeやrandom_stateは適宜調整してください
    X_train_tune, X_val_tune, Y_train_tune, Y_val_tune = train_test_split(X, Y, test_size=0.2)

    # Keras Tunerのセットアップ
    tuner = kt.RandomSearch(
        # lambdaを使って追加の引数を渡せるようにする
        lambda hp: build_model_for_tuning(
            hp, input_shape_val=X_train_tune.shape[1], output_dim_val=10
        ),
        objective="val_accuracy",  # 最適化の目標（検証データの精度）
        max_trials=10,  # 試行するハイパーパラメータの組み合わせの最大数
        executions_per_trial=1,  # 各試行でモデルを学習する回数
        directory="keras_tuner_dir",  # チューニング結果を保存するディレクトリ
        project_name="audio_classification_tuning",  # プロジェクト名
    )

    print("\nStarting hyperparameter tuning...")
    # EarlyStoppingコールバック: 検証損失が改善しなくなったら学習を早期終了
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    tuner.search(
        X_train_tune,
        Y_train_tune,
        epochs=50,  # チューニング時の最大エポック数 (EarlyStoppingで制御される)
        validation_data=(X_val_tune, Y_val_tune),
        callbacks=[early_stopping_callback],
        verbose=1,
    )  # verbose=2にするとログが簡潔になります

    # 最適なハイパーパラメータの取得
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    Hyperparameter search complete.
    Optimal units for layer 1: {best_hps.get("units1")}
    Optimal dropout for layer 1: {best_hps.get("dropout1")}
    Optimal units for layer 2: {best_hps.get("units2")}
    Optimal dropout for layer 2: {best_hps.get("dropout2")}
    Optimal learning rate: {best_hps.get("learning_rate")}
    """)

    # 全学習データでモデルを再学習
    print("\nTraining on full training data...")
    model = my_MLP(
        input_shape=X.shape[1],
        output_dim=10,
        units1=best_hps.get("units1"),
        units2=best_hps.get("units2"),
        dropout1=best_hps.get("dropout1"),
        dropout2=best_hps.get("dropout2"),
    )
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])
    history = model.fit(X, Y, batch_size=32, epochs=500, verbose=1)

    # モデル構成，学習した重みの保存
    model.save("keras_model/my_model.h5")

    # 最終モデルの学習履歴をプロット
    plot_history(history)

    # 予測結果
    predict = model.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う（accuracyと混同行列）
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        test_accuracy = accuracy_score(truth_values, predicted_values)
        plot_confusion_matrix(
            predicted_values,
            truth_values,
            title=f"Acc. {test_accuracy * 100}%",
        )
        print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
