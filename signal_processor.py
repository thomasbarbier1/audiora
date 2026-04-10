import numpy as np
from scipy.signal import butter, lfilter, decimate, firwin
import librosa
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

    def _anti_alias_filter(self, iq_samples: np.ndarray) -> np.ndarray:
        order = 129
        cutoff = (self.Fa / 2) / (self.sr / 2)
        Fc = self.bw / 2
        h = firwin(numtaps = order, cutoff = cutoff)
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

    def compute(self, iq_samples: np.ndarray, if_offset_hz: float = 0.0) -> np.ndarray:
        # DSP pipeline: antialiasing filter > decimation > frequency shift > output complex real part

        if if_offset_hz != 0.0:
            t = np.arange(len(iq_samples))
            iq_samples = iq_samples * np.exp(-1j * 2 * np.pi * if_offset_hz / self.sr * t)


        print("IQ filtering ...")
        filtered = self._anti_alias_filter(iq_samples)
        print("IQ decimation ...")
        decimated = self._decimate(filtered)
        print("IQ frequency shifting ...")
        shifted = self._freqShifter(decimated)
        print("Keeping real part ...")
        real = self._to_real(shifted)
        print("Stretching audio ...")
        stretched = self._timeStretcher(real)
        return stretched
