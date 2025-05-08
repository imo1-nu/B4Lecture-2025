"""Solution to ex2 on digital filters."""

import argparse
import wave
from pathlib import Path
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve two 1-D arrays and return the result.

    Params:
        x (ndarray): First 1-D input array
        h (ndarray): Second 1-D input array
    Returns:
        y (ndarray): Convolution result
    """
    # Designate x as the longer array
    if len(x) < len(h):
        x, h = h, x

    # Input lengths
    x_len = len(x)
    h_len = len(h)

    # Convolution slice maximum length
    slice_len_max = min(x_len, h_len)

    # Initialize output y with zeros
    y = np.zeros(x_len + h_len - 1)

    # Flip one input
    h = np.flip(h)

    for i, _ in enumerate(y):
        slice_x = slice(max(0, i - slice_len_max + 1), min(x_len, i + 1))
        slice_h = slice(max(-h_len, -i - 1), x_len - 1 - i if i >= x_len else None)

        y[i] = np.dot(x[slice_x], h[slice_h])

    return y.astype(np.int16)


def lpf(size: int, cutoff: float, samplerate: int) -> np.ndarray:
    """Return the coefficients of a low-pass filter.

    Params:
        size (int): The size of the filter
        cutoff (float): The desired cutoff frequency in Hz
        samplerate (int): The sampling rate of the relevant signal.
    Returns:
        h (ndarray): Filter coefficient array
    """
    # Cutoff frequency as fraction of sample rate for sinc equation
    cutoff = cutoff / samplerate

    # Calculate sinc filter
    h = 2 * cutoff * np.sinc(2 * cutoff * np.arange(-(size / 2), size / 2 + 1))

    # Windowing

    return h


def main(args) -> None:
    """Main routine.

    Params:
        args (Namespace): Command line arguments.
    Returns: None
    """
    # Open .wav file with builtin python interface
    with wave.open(args.file, "rb") as file:
        # Number of audio frames
        nframes = file.getnframes()
        # Sample rate
        samplerate = file.getframerate()
        # Read all frames and close file
        buffer = file.readframes(nframes)

    # Copy bytes to numpy array
    a = np.frombuffer(buffer, dtype=np.int16)

    # Calculate filter coefficients
    h = lpf(99, 1000, samplerate)

    y = convolve(a, h)

    # Write output signal to file
    wavfile.write(f"{args.file[:-4]}_filtered.wav", samplerate, y)

    # -----Plotting-----

    # Filter information
    plt.figure(1, (8, 6))

    # Frequency response
    FFT_SIZE = 1024
    H = np.fft.rfft(h, FFT_SIZE)
    H_dB = 20 * np.log10(np.abs(H) + 1e-10)
    sample_freqs = np.fft.rfftfreq(FFT_SIZE, 1 / samplerate)

    plt.subplot(211)
    plt.plot(sample_freqs, H_dB)
    plt.ylim(bottom=-90)
    plt.title("Filter frequency response")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True, linestyle="--")

    # Phase response
    plt.subplot(212)
    plt.plot(sample_freqs, np.unwrap(np.angle(H)))
    plt.title("Filter phase response")
    plt.xlabel("Frequency")
    plt.ylabel("Phase (rad)")
    plt.grid(True, linestyle="--")

    plt.tight_layout(h_pad=1.2)  # Padding to prevent text overlap

    # Spectrograms
    plt.figure(2, (8, 4))

    # Source
    plt.subplot(121)
    plt.title("Source")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    f, t, Sxx = scipy.signal.spectrogram(a, samplerate)
    plt.pcolormesh(t, f, Sxx, norm="symlog")

    # Filtered
    plt.subplot(122)
    plt.title("Filtered")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    f, t, Sxx = scipy.signal.spectrogram(y, samplerate)
    plt.pcolormesh(t, f, Sxx, norm="symlog")
    print(Sxx[:1000])

    plt.tight_layout(h_pad=1.2)  # Padding to prevent text overlap

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parsed_args = parser.parse_args()

    filepath = Path(parsed_args.file)

    if filepath.is_file():
        main(parsed_args)
    else:
        raise FileNotFoundError(f"{filepath} does not exist or is not a file.")
