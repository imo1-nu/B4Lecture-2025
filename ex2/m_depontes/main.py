""" FIR実装プログラム.

FIRフィルタの設計を行い，フィルタリングを行うプログラムを記述する．
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import pydub

# 音声ファイルの読み込み
def read_audio(path: str) -> np.ndarray:
    """音声ファイルを読み込む関数.

    指定されたパスから音声ファイルを読み込み，モノラルに変換し，numpy配列として返す．
    入力：
      path(str): 音声ファイルのパス
    出力：
      audio_data(np.ndarray).shape = (音声データ): 音声データのnumpy配列
    """
    if not os.path.exists(path):
        return np.array([])

    audio = pydub.AudioSegment.from_file(path)  # wavファイルを読み込み
    audio = audio.set_channels(1)  # モノラルに変換
    audio_data = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max  # 振幅の最大値が1になるようにスケーリング

    return audio_data

class FIR:
    """ FIRフィルタの設計とフィルタリングを行うクラス.

    メンバ関数：
        convolsion_calculate: 畳み込み計算を行う関数
        digital_filter: デジタルフィルタを設計する関数
        filtering: フィルタリングを行う関数

    属性：
        
    """
    
    def __init__(self, M, fc, fs):
        """ 初期化関数.

        引数：
            M: フィルタの次数
            fc: カットオフ周波数
            fs: サンプリング周波数
        """
        self.M = M
        self.fc = fc
        self.fs = fs
        self.omega_c = 2 * np.pi * fc / fs

    def convolution_calculate(self, x, h):
        """ 畳み込み計算を行う関数.

        引数：
            x: 入力信号
            h: フィルタ係数

        戻り値：
            y: 出力信号
        """
        y = [0 for _ in range(len(x) - len(h) + 1)]
        for i in range(len(x) - len(h) + 1):
            y[i] = 0
            for j in range(len(h)):
                y[i] += x[i + j] * h[j]
        return np.array(y)    

    def low_path_filter(self, M, fc, fs):
        """ デジタルフィルタを設計する関数.
        
        入力：
            M: フィルタの次数
            fc: カットオフ周波数
            fs: サンプリング周波数
        出力：
            h: フィルタ係数
        """
        window = np.hamming(2 * M + 1)  # ハミング窓の生成
        filter = np.zeros(2 * M + 1)  # フィルタ係数の初期化

        filter = 2 * np.pi * (fc / fs) * np.sinc(2 * fc / fs * np.arange(-M,M+1))  # LPF

        return filter * window

    def high_pass_filter(self, M, fc, fs):
        """ .
        """
        window = np.hamming(2 * M + 1)  # ハミング窓の生成
        filter = np.zeros(2 * M + 1)  # フィルタ係数の初期化
        filter = -2 * fc / fs * np.sinc(2 * fc / fs * np.arange(-M, M+1))  # HPF
        return filter * window


    def filtering(
            self,
            audio: np.ndarray,
            M: int,
    ) -> np.ndarray:
        """ フィルタリングを行う関数.

        引数：
            audio: 入力信号
            M: フィルタの次数

        戻り値：
            y: 出力信号
        """
        h = self.low_path_filter(M, self.fc, self.fs)
        y = self.convolution_calculate(audio, h)
        return y
    
def plot_filter_response(h, fs):
    """フィルタの振幅特性と位相特性をプロットする関数.

    引数：
        h: フィルタ係数
        fs: サンプリング周波数
    """
    # 周波数応答を計算
    w, H = np.fft.rfftfreq(len(h), d=1/fs), np.fft.rfft(h)

    # 振幅特性
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(H)))
    plt.title("Ampliture Response")
    plt.xlabel("freq [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid()

    # 位相特性
    plt.figure()
    plt.plot(w, np.angle(H))
    plt.title("位相特性")
    plt.xlabel("周波数 [Hz]")
    plt.ylabel("位相 [ラジアン]")
    plt.grid()

    plt.show()
    
if __name__ == "__main__":
    # FIRフィルタの設計とフィルタリングを行うクラスのインスタンスを生成
    Fs = 44100
    fir = FIR(M=64, fc=1500, fs=Fs)

    # 入力信号の生成
    audio = read_audio("C:\\Users\\arowa\\Desktop\\B4Lecture-2025\\ex2\\m_depontes\\case1-ref.wav")
    # フィルタリングを行う
    h = fir.low_path_filter(M=64, fc=1500, fs=Fs)
    plot_filter_response(h, fs=Fs)
    y = fir.filtering(audio, M=64)

    # 結果のプロット
    import scipy.signal as signal
    f,t,Sxx = signal.spectrogram(audio, fs=Fs, nperseg=512)

    plt.figure()
    plt.pcolormesh(t,f,Sxx,vmax=1e-6)
    plt.xlabel(u"時間 [sec]")
    plt.ylabel(u"周波数 [Hz]")
    # plt.colorbar()

    f,t,Sxx = signal.spectrogram(y, fs=Fs, nperseg=512)

    plt.figure()
    plt.pcolormesh(t,f,Sxx,vmax=1e-6)
    plt.xlabel(u"時間 [sec]")
    plt.ylabel(u"周波数 [Hz]")
    # plt.colorbar()
    plt.show()

    # plt.plot(audio, label='Input Signal')
    # plt.plot(y, label='Filtered Signal',alpha=0.5)
    # plt.legend()
    # plt.show()