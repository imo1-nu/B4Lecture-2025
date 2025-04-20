"""
Short-Time Fourier Transform (STFT) と Inverse STFT (ISTFT) のデモモジュール。

このモジュールでは、オーディオファイルを読み込み、STFT を計算し、
ISTFT によって信号を再構成し、元の波形、スペクトログラム、再構成波形を可視化します。
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# --- 音声読み込み ---
file_path = "audio.wav"
signal, sample_rate = sf.read(file_path)

# --- パラメータ ---
frame_size = 1024
hop_size = 512
window = np.hanning(frame_size)


def stft(x, frame_size, hop_size, window):
    """
    入力信号の短時間フーリエ変換 (STFT) を計算する関数です。

    Parameters:
        x (numpy.ndarray): 入力信号。
        frame_size (int): フレームサイズ。
        hop_size (int): フレーム間のホップサイズ。
        window (numpy.ndarray): 各フレームに適用する窓関数。

    Returns:
        numpy.ndarray: 信号の STFT を表す複素数2次元配列（形状：(frame_size/2+1, n_frames)）。
    """
    frames = []
    for i in range(0, len(x) - frame_size, hop_size):
        frame = x[i : i + frame_size] * window
        spectrum = np.fft.rfft(frame)
        frames.append(spectrum)
    return np.array(frames).T


def istft(spectrogram, frame_size, hop_size, window):
    """
    スペクトログラムから元の信号を再構成する逆短時間フーリエ変換 (ISTFT) の関数です。

    Parameters:
        spectrogram (numpy.ndarray): STFTスペクトログラム（複素数2次元配列）。
        frame_size (int): フレームサイズ。
        hop_size (int): フレーム間のホップサイズ。
        window (numpy.ndarray): STFT時に使用した窓関数。

    Returns:
        numpy.ndarray: 再構成された時系列の信号。
    """
    n_frames = spectrogram.shape[1]
    output_len = (n_frames - 1) * hop_size + frame_size
    output = np.zeros(output_len)
    window_sum = np.zeros(output_len)

    for i in range(n_frames):
        frame = np.fft.irfft(spectrogram[:, i])
        start = i * hop_size
        output[start : start + frame_size] += frame * window
        window_sum[start : start + frame_size] += window**2

    nonzero = window_sum > 1e-6
    output[nonzero] /= window_sum[nonzero]
    return output


# --- 実行 ---
spectrogram = stft(signal, frame_size, hop_size, window)
reconstructed_signal = istft(spectrogram, frame_size, hop_size, window)

# --- 書き出し（任意） ---
sf.write("reconstructed.wav", reconstructed_signal, sample_rate)

# --- 軸ラベル用 ---
time_original = np.arange(len(signal)) / sample_rate
time_reconstructed = np.arange(len(reconstructed_signal)) / sample_rate
frame_times = np.arange(spectrogram.shape[1]) * hop_size / sample_rate
freqs = np.fft.rfftfreq(frame_size, 1 / sample_rate)

# --- 描画 ---
plt.figure(figsize=(14, 8))

# --- 元の波形 ---
plt.subplot(3, 1, 1)
plt.plot(time_original, signal)
plt.title("Original Waveform")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# --- スペクトログラム ---
plt.subplot(3, 1, 2)
magnitude_db = 20 * np.log10(np.abs(spectrogram) + 1e-6)
plt.imshow(
    magnitude_db,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    extent=[frame_times[0], frame_times[-1], freqs[0], freqs[-1]],
)
plt.title("Spectrogram (dB)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")

# --- 復元された波形 ---
plt.subplot(3, 1, 3)
plt.plot(time_reconstructed, reconstructed_signal)
plt.title("Reconstructed Waveform (from ISTFT)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
