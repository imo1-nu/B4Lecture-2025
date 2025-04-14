import os
from typing import List, Tuple
import pydub
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# 音声ファイルの読み込み
def read_audio(path:str) -> np.ndarray:
    '''
    入力：音声ファイルのパス
    出力：音声データのnumpy配列
    '''
    if not os.path.exists(path):
        return np.array([])
    
    audio = pydub.AudioSegment.from_file(path) # 音声ファイルの読み込み
    audio_data = audio.get_array_of_samples() # 音声ファイルからデータの取得
    
    return np.array(audio_data)

# 音声データの切り出し
def cut_audio(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    '''
    入力：音声データのnumpy配列、切り出す開始位置、終了位置
    出力：切り出した音声データのnumpy配列
    '''
    if start < 0 or end > len(audio) or start >= end:
        raise ValueError("Invalid start or end indices")
    
    return audio[start:end]

# 音声データに窓関数を適用
def window_function(audio: np.ndarray) -> np.ndarray:
    '''
    入力：音声データのnumpy配列
    出力：窓関数を適用した音声データのnumpy配列
    '''
    return np.hanning(len(audio)) * audio

# FFTを計算
def fft(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    入力：音声データのnumpy配列
    出力：FFTの結果と周波数のnumpy配列
    '''
    if len(audio) == 0:
        return np.array([]), np.array([])
    
    fft_result = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio))
    
    return fft_result, freqs

# スペクトログラムを作成
def make_spectrogram(fft_result: np.ndarray, freqs: np.ndarray) -> None:
    '''
    入力：FFTの結果と周波数のnumpy配列
    出力：なし（スペクトログラムを表示）
    '''
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(fft_result), aspect='auto', extent=[0, len(fft_result), min(freqs), max(freqs)], origin='lower')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (samples)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()

    plt.mlab.specgram(fft_result, Fs=44100, NFFT=1024, noverlap=512, cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()
    