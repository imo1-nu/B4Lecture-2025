"""
FIRフィルタを用いた音声信号処理モジュール。

このスクリプトでは、LPF・HPF・BPF・BEFのFIRフィルタを実装し、
音声ファイルに対して適用し、スペクトログラムおよびフィルタ特性を出力します。
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
import scipy.signal as signal


def read_audio(path: str) -> tuple[int, np.ndarray]:
    """音声ファイルを読み込み、モノラル・正規化を行う。

    Args:
        path (str): 音声ファイルのパス。

    Returns:
        tuple[int, np.ndarray]: サンプリング周波数と正規化された音声データ。
    """
    if not os.path.exists(path):
        return 0, np.array([])

    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # モノラル化
    data = data.astype(np.float32) / np.iinfo(data.dtype).max  # 正規化
    return fs, data


def save_audio(path: str, data: np.ndarray, fs: int):
    """正規化された音声データを16bit整数に変換し、WAVファイルとして保存する。

    Args:
        path (str): 保存先パス。
        data (np.ndarray): 音声データ。
        fs (int): サンプリング周波数。
    """
    scaled = np.int16(np.clip(data, -1.0, 1.0) * 32767)
    wavfile.write(path, fs, scaled)


class FIR:
    """FIRフィルタの生成と畳み込みを行うクラス。"""

    def __init__(self, M, fc, fs, fc2=None):
        """FIRフィルタの初期化。

        Args:
            M (int): フィルタの半分の長さ（遅延）。
            fc (float): カットオフ周波数1。
            fs (int): サンプリング周波数。
            fc2 (float, optional): カットオフ周波数2（バンドパスやバンドストップ用）。
        """
        self.M = M
        self.fc = fc
        self.fc2 = fc2
        self.fs = fs

    def convolution_calculate(self, x, h):
        """畳み込み計算を自前で実装する。

        Args:
            x (np.ndarray): 入力信号。
            h (np.ndarray): フィルタ係数。

        Returns:
            np.ndarray: 畳み込み出力。
        """
        y = np.zeros(len(x) - len(h) + 1)
        for n in range(len(y)):
            y[n] = np.sum(x[n : n + len(h)] * h)
        return y

    def low_pass_filter(self, fc):
        """ローパスフィルタのインパルス応答を生成する。

        Args:
            fc (float): カットオフ周波数。

        Returns:
            np.ndarray: フィルタ係数。
        """
        omega = 2 * fc / self.fs
        n = np.arange(-self.M, self.M + 1)
        h = omega * np.sinc(omega * n)
        return h

    def high_pass_filter(self):
        """ハイパスフィルタのインパルス応答を生成する。

        Returns:
            np.ndarray: フィルタ係数。
        """
        omega = 2 * self.fc / self.fs
        n = np.arange(-self.M, self.M + 1)
        h = np.sinc(n) - omega * np.sinc(omega * n)
        return h

    def band_pass_filter(self):
        """バンドパスフィルタのインパルス応答を生成する。

        Returns:
            np.ndarray: フィルタ係数。
        """
        low = self.low_pass_filter(self.fc2)
        high = self.low_pass_filter(self.fc)
        h = low - high
        return h

    def band_stop_filter(self):
        """バンドストップフィルタのインパルス応答を生成する。

        Returns:
            np.ndarray: フィルタ係数。
        """
        h = np.sinc(np.arange(-self.M, self.M + 1)) - self.band_pass_filter()
        return h

    def filtering(self, audio: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """ハミング窓を適用してフィルタ処理を行う。

        Args:
            audio (np.ndarray): 入力音声。
            filt (np.ndarray): フィルタ係数。

        Returns:
            np.ndarray: フィルタ後の音声。
        """
        window = np.hamming(len(filt))
        return self.convolution_calculate(audio, filt * window)


def plot_filter_response(h: np.ndarray, fs: int, title: str, folder: str):
    """フィルタの振幅・位相特性をプロットし、画像保存する。

    Args:
        h (np.ndarray): フィルタ係数。
        fs (int): サンプリング周波数。
        title (str): グラフのタイトル。
        folder (str): 保存先フォルダ。
    """
    N = 512
    H = np.fft.rfft(h, n=N)
    w = np.fft.rfftfreq(N, d=1 / fs)

    # 1行2列のサブプロットを作成
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # 振幅特性
    axs[0].plot(w, 20 * np.log10(np.abs(H) + 1e-10))
    axs[0].set_title(f"Amplitude Response of {title}")
    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].set_ylabel("Amplitude [dB]")
    axs[0].set_xlim([0, 8000])
    axs[0].grid()

    # 位相特性
    axs[1].plot(w, np.unwrap(np.angle(H)))
    axs[1].set_title(f"Phase Response of {title}")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Phase [radian]")
    axs[1].set_xlim([0, 8000])
    axs[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{title}_response.png"))
    plt.show()


def make_spectrogram(audio: np.ndarray, Fs: int, title: str, folder: str):
    """音声のスペクトログラムを作成・保存する。

    Args:
        audio (np.ndarray): 音声データ。
        Fs (int): サンプリング周波数。
        title (str): グラフのタイトル。
        folder (str): 保存先フォルダ。
    """
    f, t, Sxx = signal.spectrogram(audio, fs=Fs, nperseg=512)
    plt.figure()
    plt.pcolormesh(t, f, Sxx, shading="gouraud", vmax=1e-6)
    plt.xlabel("time [sec]")
    plt.ylabel("frequency [Hz]")
    plt.title(title)
    plt.colorbar(label="Power")
    plt.ylim([0, 8000])
    plt.savefig(os.path.join(folder, f"{title}_spectrogram.png"))
    plt.show()


if __name__ == "__main__":
    input_path = "audio.wav"
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    M = 128
    fc = 3000
    fc2 = 6000

    fs, audio = read_audio(input_path)
    if audio.size == 0:
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {input_path}")

    fir = FIR(M=M, fc=fc, fc2=fc2, fs=fs)

    make_spectrogram(audio, fs, "Original Audio Spectrogram", output_folder)

    filters = [
        (fir.low_pass_filter(fc), "Low-Pass Filter", "filtered_low.wav"),
        (fir.high_pass_filter(), "High-Pass Filter", "filtered_high.wav"),
        (fir.band_pass_filter(), "Band-Pass Filter", "filtered_bandpass.wav"),
        (fir.band_stop_filter(), "Band-Stop Filter", "filtered_bandstop.wav"),
    ]

    for filter_coeff, title, output_name in filters:
        filtered = fir.filtering(audio, filter_coeff)

        plot_filter_response(filter_coeff, fs, title, output_folder)
        make_spectrogram(
            filtered, fs, f"Filtered Audio Spectrogram ({title})", output_folder
        )

        save_audio(output_name, filtered, fs)
        print(f"フィルタ後の音声を保存しました: {output_name}")
