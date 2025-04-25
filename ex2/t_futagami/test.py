import numpy as np
from matplotlib import pyplot as plt

N = 1024
sr = 16000

# f_l=1000, f_h=4500のBEFの長さNの振幅特性
f_l = 1000
f_h = 4500
freqs = np.fft.fftfreq(N, d=1 / sr)  # 周波数軸 [0, f_max] → [-f_max, 0]
H = np.zeros(N)
for i in range(N):
    if f_l <= abs(freqs[i]) <= f_h:
        H[i] = 1

h = np.fft.ifft(H)
M = 128
# 右にM / 2だけシフト
h = np.roll(h, M // 2)
for i in range(N):
    if i > M:
        h[i] = 0

Filter = np.fft.fft(h.real)  # 時間領域から周波数領域に変換
amp_dB = 20 * np.log10(abs(Filter))  # 振幅特性をdBに変換
deg = np.unwrap(np.angle(Filter))  # 位相特性

# 振幅特性のプロット
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(freqs[: N // 2], amp_dB[: N // 2])  # 周波数軸は[0, f_max]にする
plt.xlim(0, sr / 2)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Amplitude characteristic")

# 位相特性のプロット
plt.subplot(2, 1, 2)
plt.plot(freqs[: N // 2], deg[: N // 2])  # 位相特性
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [rad]")
plt.title("Phase characteristic")

plt.tight_layout()
plt.show()
