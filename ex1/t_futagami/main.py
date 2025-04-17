from matplotlib import pyplot as plt
import numpy as np
import soundfile as sf
import librosa

def stft (audio_data: np.array, window_len: int, overlap_rate: float):
    spec = []
    window_func = np.hamming(window_len)
    step = window_len * (1 - overlap_rate)
    for i in range(0, len(audio_data) - window_len, int(step)):
        frame = audio_data[i:i + window_len] * window_func
        spectrum = np.fft.fft(frame)
        spectrum = spectrum[spectrum.shape[0] // 2:]
        spec.append(spectrum)
    print(f"fft.size: {len(spec[0])}")
    return np.array(spec)

def istft (spec: np.array, window_len: int, overlap_rate: float, original_len: int):
    window_func = np.hamming(window_len)
    step = window_len * (1 - overlap_rate)
    audio_data = np.zeros(int(len(spec) * step + window_len))
    for i in range(0, len(spec)):
        spectrum = np.concatenate((spec[i], np.conj(spec[i][::-1])))
        ifft_frame = np.fft.ifft(spectrum)
        ifft_frame = ifft_frame / window_func
        start = int(i * step)
        audio_data[start:start + window_len] += ifft_frame.real
    return audio_data[:original_len]

def main():
    data, sample_rate = sf.read("audio.wav")
    nyquist_freq = sample_rate // 2
    window_len = 1024
    overlap_rate = 0.5
    spec = stft(data, window_len, overlap_rate)
    audio_reconstructed = istft(spec, window_len, overlap_rate, len(data))
    spec_dB = 20 * np.log10(np.abs(spec))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5)

    x = np.linspace(0, len(data) / sample_rate, len(data))
    axes[0].plot(x, data)
    axes[0].set_title("Original Signal")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    im = axes[1].imshow(
        spec_dB.T,
        cmap="jet",
        aspect="auto",
        vmin=-60,
        vmax=30,
        extent=[0, len(data) / sample_rate, 0, nyquist_freq],
    )
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlim(0, len(data) / sample_rate)
    axes[1].set_ylim(0, nyquist_freq)
    plt.colorbar(mappable=im, ax=axes[1])

    axes[2].plot(x, audio_reconstructed)
    axes[2].set_title("Reconstructed Signal")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    sf.write("audio_reconstructed.wav", audio_reconstructed, sample_rate)

main()