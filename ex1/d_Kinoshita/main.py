"""
STFTとISTFTによる音声信号の解析と再構成を行うスクリプト。
音声ファイルを読み込み、スペクトログラムを表示し、逆変換して波形を復元します。
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


# --- STFT ---
def stft(x, frame_size, hop_size, window):
    """
    短時間フーリエ変換（Short-Time Fourier Transform）を行う関数。

    Parameters:
        x (np.ndarray): 入力信号
        frame_size (int): フレームサイズ
        hop_size (int): フレーム間隔
        window (np.ndarray): 窓関数

    Returns:
        np.ndarray: スペクトログラム（周波数×時間）
    """
    frames = []
    for i in range(0, len(x) - frame_size, hop_size):
        frame = x[i : i + frame_size] * window
        spectrum = np.fft.rfft(frame)
        frames.append(spectrum)
    return np.array(frames).T


# --- 逆STFT ---
def istft(spectrogram, frame_size, hop_size, window):
    """
    逆短時間フーリエ変換（Inverse STFT）を行う関数。

    Parameters:
        spectrogram (np.ndarray): STFT で得られたスペクトログラム
        frame_size (int): フレームサイズ
        hop_size (int): フレーム間隔
        window (np.ndarray): 窓関数

    Returns:
        np.ndarray: 復元された波形
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
