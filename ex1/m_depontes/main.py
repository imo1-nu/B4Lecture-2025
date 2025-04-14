from ex1.m_depontes.process_audio import *

def main():
    shift_length = 1000
    audio = read_audio("audio.wav")
    for n in range(0,len(audio)-shift_length, 800):
        cut_audio = cut_audio(audio, n, n+shift_length)
        windowed_audio = window_function(cut_audio)
        fft_result, freqs = fft(windowed_audio)
        make_spectrogram(fft_result, freqs)
