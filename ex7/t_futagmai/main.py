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

import librosa
import matplotlib.pyplot as plt
import numpy as np
import optuna
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


def extract_mfcc(data, n_mfcc=100):  # n_mfcc は基本MFCC係数の数
    """
    音声データからMFCCとそのデルタ、ダブルデルタ特徴量を抽出する
    MFCC、デルタMFCC、ダブルデルタMFCCを計算し、時間軸で平均化
    これら3種の特徴量を連結し、1次元の特徴ベクトルとする

    Args:
        data: 音声波形データのリスト (各要素はNumPy配列)
        n_mfcc: 抽出する基本MFCC係数の数 (例: 13, 20, 100)
    Returns:
        features_array: 抽出された特徴量のNumPy配列 (形状: num_samples, n_segments * 3 * n_mfcc)
    """
    features_list = []
    for audio_signal in data:
        mfcc_orig = librosa.feature.mfcc(y=audio_signal, n_mfcc=n_mfcc)
        # デルタ特徴量を計算 (入力と同じ形状でパディングされて返される)
        delta_mfcc = librosa.feature.delta(mfcc_orig, width=5)
        delta2_mfcc = librosa.feature.delta(mfcc_orig, width=5, order=2)

        mean_mfcc_orig = np.mean(mfcc_orig, axis=1)
        mean_delta_mfcc = np.mean(delta_mfcc, axis=1)
        mean_delta2_mfcc = np.mean(delta2_mfcc, axis=1)

        # このセグメントの平均化された特徴量を連結
        feature_vec = np.concatenate((mean_mfcc_orig, mean_delta_mfcc, mean_delta2_mfcc))
        features_list.append(feature_vec)

    return np.array(features_list)


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

    train_features = extract_mfcc(train_data)
    test_features = extract_mfcc(test_data)

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


def objective(trial, X_train, Y_train, X_val, Y_val, input_shape, output_dim):
    """
    Optunaの目的関数
    Args:
        trial: OptunaのTrialオブジェクト
        X_train, Y_train: 学習データ
        X_val, Y_val: 検証データ
        input_shape: 入力データの形状
        output_dim: 出力層の次元数
    Returns:
        検証データの精度 (最大化を目指す)
    """
    model = Sequential()

    # ハイパーパラメータの提案
    units1 = trial.suggest_int("units1", 64, 512, step=64)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.5, step=0.1)
    units2 = trial.suggest_int("units2", 64, 512, step=64)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])

    model.add(Dense(units=units1, activation="relu", input_dim=input_shape))
    model.add(Dropout(rate=dropout1))
    model.add(Dense(units=units2, activation="relu"))
    model.add(Dropout(rate=dropout2))
    model.add(Dense(output_dim, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # EarlyStoppingコールバック
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        Y_train,
        epochs=50,  # チューニング時の最大エポック数
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping_callback],
        verbose=0,  # Optunaの試行中はログを抑制することが多い
    )

    # 検証データの精度を取得 (最後の値、またはEarlyStoppingで最良だった値)
    val_accuracy = history.history["val_accuracy"][
        -1
    ]  # もしrestore_best_weights=Trueならこれが最良
    # もしくは、より確実に最良のval_accuracyを取得する場合:
    # val_accuracy = np.max(history.history['val_accuracy'])
    return val_accuracy


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

    print("\nStarting hyperparameter tuning with Optuna...")
    # OptunaのStudyオブジェクトを作成
    # direction="maximize" で目的関数の戻り値を最大化する
    study = optuna.create_study(direction="maximize")

    # 最適化の実行
    # lambdaを使って固定引数を渡す
    study.optimize(
        lambda trial: objective(
            trial,
            X_train_tune,
            Y_train_tune,
            X_val_tune,
            Y_val_tune,
            input_shape=X_train_tune.shape[1],
            output_dim=10,
        ),
        n_trials=10,  # 試行するハイパーパラメータの組み合わせの最大数
        # n_jobs=-1 # 並列実行する場合 (環境によっては設定)
    )

    # 最適なハイパーパラメータの取得
    best_hps = study.best_params

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
