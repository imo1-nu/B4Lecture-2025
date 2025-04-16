import os
from typing import List, Tuple
import pydub
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# 音声ファイルの読み込み
def read_audio(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.array([])

    audio = pydub.AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio_data = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_data /= np.iinfo(np.int16).max  # 振幅の最大値が1になるようにスケーリング

    return audio_data

class stft:
  def __init__(self, audio: np.ndarray, frame_length: int, frame_shift: int, sample_rate: int):
    self.audio = audio
    self.frame_length = frame_length
    self.frame_shift = frame_shift
    self.sample_rate = sample_rate

    self.frames = self.cut_audio()
    self.windowed_frames = self.window_function()
    self.spectrogram = self.fft()
    self.times = np.arange(len(self.spectrogram)) * self.frame_shift / self.sample_rate
    self.freqs = np.fft.fftfreq(self.frame_length, d=1/self.sample_rate)[:self.frame_length//2]


  # 音声データの切り出し
  def cut_audio(self) -> List[np.ndarray]:
      '''
      入力：音声データのnumpy配列、切り出す開始位置、終了位置
      出力：切り出した音声データのnumpy配列
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
      出力：窓関数を適用した音声データのnumpy配列
      '''
      return [np.hanning(len(audio)) * audio for audio in self.frames]

  # FFTを計算
  def fft(self) -> List[np.ndarray]:
      '''
      入力：音声データのnumpy配列
      出力：FFTの結果と周波数のnumpy配列
      '''
      if len(self.windowed_frames) == 0:
          return []

      return np.array([np.abs(np.fft.fft(frames)[:self.frame_length//2]) for frames in self.windowed_frames])

###

class istft:
  def __init__(self, audio: np.ndarray, frame_length: int, frame_shift: int, sample_rate: int):
    self.audio = audio
    self.frame_length = frame_length
    self.frame_shift = frame_shift
    self.sample_rate = sample_rate

    self.frames = self.cut_audio()
    self.windowed_frames = self.window_function()
    self.spectrogram = self.fft()
    self.times = np.arange(len(self.spectrogram)) * self.frame_shift / self.sample_rate
    self.freqs = np.fft.fftfreq(self.frame_length, d=1/self.sample_rate)[:self.frame_length//2]


  # 音声データの切り出し
  def cut_audio(self) -> List[np.ndarray]:
      '''
      入力：音声データのnumpy配列、切り出す開始位置、終了位置
      出力：切り出した音声データのnumpy配列
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
      出力：窓関数を適用した音声データのnumpy配列
      '''
      return [np.hanning(len(audio)) * audio for audio in self.frames]

  # FFTを計算
  def ifft(self) -> List[np.ndarray]:
      '''
      入力：音声データのnumpy配列
      出力：FFTの結果と周波数のnumpy配列
      '''
      if len(self.windowed_frames) == 0:
          return []

      return np.array([np.abs(np.fft.ifft(frames)[:self.frame_length//2]) for frames in self.windowed_frames])

### 

def plot_waveform(audio: np.ndarray, sample_rate: int = 44100) -> None:
  """
  入力：音声データのnumpy配列、サンプリングレート
  出力：なし（波形を表示）
  """
  time = np.arange(0, len(audio)) / sample_rate
  plt.figure(figsize=(10, 4))
  plt.plot(time, audio)
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.title('Waveform')
  plt.grid(True)
  plt.show()

# スペクトログラムを作成
def make_spectrogram(spectrogram: np.ndarray, times, freqs) -> None:
    plt.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[0, times[-1], freqs[0], freqs[-1]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()

def main():
    audio = read_audio("case1-ref.wav")
    plot_waveform(audio)

    sample_rate = 44100
    frame_length = 1024
    frame_shift = 512

    spectrogram = []
    stft_instance = stft(audio, frame_length, frame_shift, sample_rate)

    spectrogram, times, freqs = stft_instance.spectrogram, stft_instance.times, stft_instance.freqs

    make_spectrogram(spectrogram, times, freqs)

main()