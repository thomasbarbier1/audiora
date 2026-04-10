import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import csv
from pathlib import Path
from typing import Iterable, List

def show_radio_output(iq_chunk, radio_sr, title):
    t = np.arange(len(iq_chunk)) / radio_sr
    signal = np.abs(iq_chunk)
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig(f"{title}.jpeg", dpi=300, bbox_inches="tight")
    plt.show()

def show_spectrogram(iq, title):
    f, t, Sxx = spectrogram(iq, fs=1.0, return_onesided=False)
    plt.imshow(10 * np.log10(np.abs(Sxx) + 1e-12), aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(f"{title}.jpeg", dpi=300, bbox_inches="tight")
    plt.show()

def show_audioSignal(signal, samplerate, title='Resulting signal (time+FFT)'):
    n = len(signal)
    t = np.arange(n) / samplerate

    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    X = np.fft.rfft(signal)
    X = np.abs(X)
    f = np.fft.rfftfreq(n, d=1.0 / samplerate)

    plt.figure(figsize=(10, 6))

    # Signal temporel
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Time domain")
    plt.grid(True)

    # FFT magnitude
    plt.subplot(2, 1, 2)
    plt.plot(f, X)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|FFT|")
    plt.title("Frequency domain (FFT magnitude)")
    plt.grid(True)

    plt.savefig(f"{title}.jpeg", dpi=300, bbox_inches="tight")
    plt.show()

def show_psd(samples, fs, fc):
    # use matplotlib to estimate and plot the PSD
    plt.psd(samples, Fs=fs, Fc=fc)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Relative power (dB)')
    plt.show()

def export_to_csv(data: Iterable[complex],filename: str) -> None:

    file_path = Path.cwd() / filename

    with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        for z in data:
            writer.writerow([z.real, z.imag])

def load_from_csv(filename: str) -> List[complex]:
    """
    Charge un fichier CSV contenant des nombres complexes et retourne
    une liste de complexes.

    - Répertoire de lecture : racine du projet (cwd)
    - Délimiteur attendu : point-virgule (;)
    - Format par ligne : real;imag
    """
    file_path = Path.cwd() / filename
    complex_array = []

    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            real = float(row[0])
            imag = float(row[1])
            complex_array.append(complex(real, imag))

    return np.array(complex_array)