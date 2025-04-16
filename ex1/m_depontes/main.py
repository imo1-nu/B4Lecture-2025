import os
from typing import List
import pydub
import numpy as np
import matplotlib.pyplot as plt

# 音声ファイルの読み込み
def read_audio(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.array([])

    audio = pydub.AudioSegment.from_file(path) # wavファイルを読み込み
    audio = audio.set_channels(1) # モノラルに変換
    audio_data = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max  # 振幅の最大値が1になるようにスケーリング

    return audio_data

# stft実装クラス
class stft:
  def __init__(self, audio: np.ndarray, frame_length: int, frame_shift: int, sample_rate: int):
    self.audio = audio
    self.frame_length = frame_length
    self.frame_shift = frame_shift
    self.sample_rate = sample_rate

    # 与えたaudioに対して切り出し・窓関数の付加・fftの実行をする
    self.frames = self.cut_audio()
    self.windowed_frames = self.window_function()
    self.spectrogram = self.fft()
    self.times = np.arange(len(self.spectrogram)) * self.frame_shift / self.sample_rate
    self.freqs = np.fft.fftfreq(self.frame_length, d=1/self.sample_rate)[:self.frame_length//2]

  # 音声データの切り出し
  def cut_audio(self) -> List[np.ndarray]:
      '''
      入力：音声データのnumpy配列、切り出す開始位置、終了位置
      出力：切り出した音声データのnumpy配列の集合
      '''
      frames = []
      for n in range(0,len(self.audio)-self.frame_length, self.frame_shift):
        frame = self.audio[n:n+self.frame_length]
        frames.append(frame)

      return frames

  # 音声データに窓関数を適用
  def window_function(self) -> List[np.ndarray]:
      '''
      入力：音声データのnumpy配列
      出力：窓関数を適用した音声データのnumpy配列の集合
      '''
      return [np.hanning(len(audio)) * audio for audio in self.frames]

  # FFTを計算
  def fft(self) -> np.ndarray:
      '''
      入力：音声データのnumpy配列
      出力：FFTの結果と周波数のnumpy配列
      '''
      if len(self.windowed_frames) == 0:
          return []

      return np.array([np.fft.fft(frame) for frame in self.windowed_frames])


# ISTFT実装クラス
class istft:
  def __init__(self, spectrogram: np.ndarray, frame_length: int, frame_shift: int, sample_rate: int):
    self.spectrogram = spectrogram
    self.frame_length = frame_length
    self.frame_shift = frame_shift
    self.sample_rate = sample_rate

    self.frames = self.ifft()
    self.output_len = (len(self.frames) - 1) * self.frame_shift + self.frame_length
    self.reconstructed_wave = self.reconstruct_wave()
    
  # iFFTを計算
  def ifft(self) -> List[np.ndarray]:
      '''
      入力：スペクトログラムのnumpy配列
      出力：iFFTの結果のnumpy配列
      '''
      if len(self.spectrogram) == 0:
        return []

      return np.array([np.fft.ifft(spec).real for spec in self.spectrogram])

  # 波形の再構築
  def reconstruct_wave(self) -> List[np.ndarray]:
    '''
    入力：音声データのnumpy配列、切り出す開始位置、終了位置
    出力：切り出した音声データのnumpy配列
    '''
    num_frames = len(self.spectrogram) # フレーム数
    reconstructed = np.zeros(self.output_len) # 再構築する音声データの枠
    window_sum = np.zeros(self.output_len) # 補正する窓関数の合計値
    window = np.hanning(self.frame_length) # 窓関数（ハミング関数）の定義

    # 各フレームに対して、窓関数を適用し、再構築する
    for i in range(num_frames):
        start = i * self.frame_shift # フレームの開始位置
        reconstructed[start:start + self.frame_length] += self.frames[i] * window # 窓関数によって重みづけ
        window_sum[start:start + self.frame_length] += window ** 2 # 窓関数の合計値を計算

        nonzero = window_sum > 1e-10 # 窓関数の合計値が0である(=窓関数によって歪曲されてない)箇所以外を取得
        reconstructed[nonzero] /= window_sum[nonzero] # 窓関数の合計値で割ることで、歪曲された部分を補正する

    return reconstructed
        
###

# 波形をプロットする関数
def plot_waveform(audio: np.ndarray, title:str, file_name:str, sample_rate: int = 44100) -> None:
  """
  入力：音声データのnumpy配列、サンプリングレート
  出力：なし（波形を表示）
  """
  time = np.arange(0, len(audio)) / sample_rate
  plt.figure(figsize=(10, 4))
  plt.plot(time, audio)
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.title(title)
  plt.grid(True)
  plt.show()
  plt.savefig(file_name) # 波形を保存

# スペクトログラムを作成
def make_spectrogram(spectrogram: np.ndarray, times, freqs) -> None:
    '''
    入力：スペクトログラムのnumpy配列、時間軸、周波数軸
    出力：なし（スペクトログラムを表示）
    '''
    plt.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[0, times[-1], freqs[0], freqs[-1]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()
    plt.savefig() # スペクトログラムを保存

# メイン関数
def main():
    audio = read_audio("case1-ref.wav") # wavファイルを読み込み
    plot_waveform(audio,"original wave", "original_wave") # 波形をプロット

    sample_rate = 44100
    frame_length = 1024
    frame_shift = 512
    spectrogram = []

    # STFTを計算・スペクトログラムを作成
    stft_instance = stft(audio, frame_length, frame_shift, sample_rate)
    spectrogram, times, freqs = stft_instance.spectrogram, stft_instance.times, stft_instance.freqs

    make_spectrogram(np.abs(spectrogram[:frame_length//2]), times, freqs) 
    # [:frame_length//2]でスペクトログラムのうち正の周波数成分のみを切り出し

    # ISTFTを計算・波形を再構築
    istft_instance = istft(spectrogram, frame_length, frame_shift, sample_rate)
    reconstructed_wave = istft_instance.reconstructed_wave

    plot_waveform(reconstructed_wave, "reconstructed wave", "reconstructed_wave") # 再構築した波形をプロット

main()