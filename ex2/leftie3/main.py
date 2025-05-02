"""Solution to ex2 on digital filters."""

import argparse
import wave
from pathlib import Path
import numpy as np


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve two 1-D arrays and return the result

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


def main(args) -> None:
    """Main routine.

    Params:
        args (Namespace): Command line arguments.
    Returns: None
    """
    # Open .wav file with builtin python interface
    with wave.open(args.file, "rb") as file:
        # Get number of audio frames
        nframes = file.getnframes()
        # Read all frames and close file
        buffer = file.readframes(nframes)

    # Copy bytes to numpy array
    a = np.frombuffer(buffer, dtype=np.int16)

    h = np.array([1, 2, 3, 3, 3])

    y = np.convolve(a[:100], h)
    print(y)
    y = convolve(a[:100], h)
    print(y)


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    filepath = Path(args.file)

    if filepath.is_file():
        main(args)
    else:
        raise FileNotFoundError(f"{filepath} does not exist or is not a file.")
