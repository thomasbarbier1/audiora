import numpy as np
from scipy.signal import butter, lfilter, decimate, firwin, resample_poly
import librosa
from utils import *
from typing import Iterable

def downsampler(signal: np.ndarray, up, down):
    return resample_poly(signal, up, down, window=('kaiser', 8.0))

"""
class LoRaMapping:
    _ch1: float = 868_100_000.
    _ch2: float = 863_300_000.
    _ch3: float = 868_500_000.
    LoraLow_ch1: float = _ch1 - (125_000. / 2)
    LoraHigh_ch1: float = _ch1 + (125_000. / 2)
    LoraLow_ch2: float = _ch2 - (125_000. / 2)
    LoraHigh_ch2: float = _ch2 + (125_000. / 2)
    LoraLow_ch3: float = _ch3 - (125_000. / 2)
    LoraHigh_ch3: float = _ch3 + (125_000. / 2)
    # Offsets
    ch1_offset: float = _ch1 - AudioMid
    ch2_offset: float = _ch2 - AudioMid
    ch3_offset: float = _ch3 - AudioMid
"""

class AudioMapping:
    Lo: float = 220.                # audio band lowest frequency
    Hi: float = 2_200.              # audio band highest frequency
    Fc: float = (Hi + Lo) / 2.      # audio band central frequency
    Bw: float = Hi - Lo             # audio bandwidth

class SignalProcessor:

    def __init__(self, sr: float, bw, stretching_factor: int = 4):
        self.sr = sr
        self.bw = bw
        self.scaling_factor = int(self.sr / AudioMapping.Bw)
        self.stretching_factor = stretching_factor
        self.Fa = self.sr / self.stretching_factor

    def truncate(self, signal: np.ndarray) -> np.ndarray:
        nb_samples_to_cut = int(self.sr * 0.01)
        if len(signal) < 10 * nb_samples_to_cut:
            return signal
        return signal[nb_samples_to_cut:-nb_samples_to_cut]

    @staticmethod
    def normalize_complex_magnitude(signal: Iterable[complex]) -> np.ndarray:
        """
        Normalise le module d’un tableau de nombres complexes sur [0,1]
        en conservant la phase, puis retourne le signal reconstruit en a + jb.
        """
        z = np.asarray(signal, dtype=complex)

        # Module et phase
        magnitude = np.abs(z)
        phase = np.angle(z)

        # Normalisation des modules (min–max)
        mag_min = magnitude.min()
        mag_max = magnitude.max()

        if mag_max == mag_min:
            mag_norm = np.zeros_like(magnitude)
        else:
            mag_norm = (magnitude - mag_min) / (mag_max - mag_min)

        # Reconstruction : mag_norm * exp(j * phase)
        z_normalized = mag_norm * np.exp(1j * phase)

        return z_normalized

    @staticmethod
    def _anti_alias_filter(iq_samples: np.ndarray, cutoff, order = 129) -> np.ndarray:
        h = firwin(order, cutoff)
        return lfilter(h, 1.0, iq_samples)

    def _decimate(self, iq_samples: np.ndarray) -> np.ndarray:
        return iq_samples[::self.scaling_factor]

    def _freqShifter(self, iq_samples: np.ndarray) -> np.ndarray:
        t = np.arange(len(iq_samples))
        return iq_samples * np.exp(1j * 2 * np.pi * AudioMapping.Fc * t / self.Fa)

    def _to_real(self, iq_samples: np.ndarray) -> np.ndarray:
        return np.real(iq_samples)

    def _timeStretcher(self, audio: np.ndarray) -> np.ndarray:
        return librosa.effects.time_stretch(audio, rate=1/self.stretching_factor)

    def preProcess(self, iq_samples: np.ndarray) -> np.ndarray:
        truncated = self.truncate(iq_samples)
        normalized = self.normalize_complex_magnitude(truncated)
        return normalized

    def compute(self, iq_samples: np.ndarray) -> np.ndarray:

        # DSP pipeline: scaling > frequency shift > output complex real part

        # 1) Shifting
        # The input BW is 125kHz wide, and we want it to be ~2kHz
        # This is done by downsampling the signal with a factor 62.5 (or here, 63 because the function needs an integer)
        print("Processing: downsampling to fit audio bandwidth ...")
        scaled = downsampler(iq_samples, up=1, down=63)

        print("Processing: shifting to fit audio central frequency ... ")
        shifted = self._freqShifter(scaled)

        print("Process: keeping real part ...")
        real = self._to_real(shifted)

        # print("Stretching audio ...")
        # stretched = self._timeStretcher(real)

        return real
