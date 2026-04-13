"""
LoRa IQ -> audible dsp pipeline.

Core strategy:
1) baseband cleanup
2) instantaneous frequency estimation from the complex phase derivative
3) robust smoothing and centering
4) frequency mapping into an audible band
5) time stretching for easier listening
6) synthesis of a mono waveform (with subtle clipping)

The default parameters are chosen to make LoRa chirps clearly audible while
keeping the output in a conventional audio range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, savgol_filter


@dataclass
class Dsp:
    """Offline DSP chain for LoRa IQ sonification."""

    input_fs: float = 2_400_000.0
    lora_bw: float = 125_000.0

    # Audio output settings
    audio_fs: int = 48_000
    audio_center_hz: float = 2000.0
    audio_span_hz: float = 1000.0
    time_stretch: float = 1.0

    # Pre-processing
    enable_lowpass: bool = True
    lowpass_margin_hz: float = 8_000.0
    lowpass_order: int = 8

    # Instantaneous frequency smoothing
    smooth_ms: float = 0.35
    use_savgol: bool = True
    savgol_polyorder: int = 2

    # Frequency estimation robustness
    amplitude_floor: float = 1e-8
    robust_percentile: float = 99.5
    remove_dc_offset: bool = True

    # Output shaping
    fade_ms: float = 20.0
    output_gain: float = 1.0
    soft_clip: bool = False
    soft_clip_drive: float = 1.3

    # Internal state for debugging / inspection
    last_debug: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.input_fs <= 0:
            raise ValueError("input_fs must be > 0")
        if self.audio_fs <= 0:
            raise ValueError("audio_fs must be > 0")
        if self.lora_bw <= 0:
            raise ValueError("lora_bw must be > 0")
        if self.time_stretch <= 0:
            raise ValueError("time_stretch must be > 0")
        if self.audio_span_hz <= 0:
            raise ValueError("audio_span_hz must be > 0")
        if self.audio_center_hz <= 0:
            raise ValueError("audio_center_hz must be > 0")
        if self.audio_center_hz + self.audio_span_hz >= 0.48 * self.audio_fs:
            raise ValueError(
                "audio_center_hz + audio_span_hz must stay comfortably below Nyquist"
            )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def process(self, iq: np.ndarray, *, return_debug: bool = False) -> np.ndarray:
        """Convert complex IQ samples into a mono audible waveform.

        Parameters
        ----------
        iq:
            1-D complex numpy array.
        return_debug:
            If True, debug arrays are stored in `self.last_debug`.

        Returns
        -------
        np.ndarray
            Float32 mono waveform in approximately [-1, 1].
        """
        x = self._validate_iq(iq)
        x = self._normalize_complex_amplitude(x)

        if self.enable_lowpass:
            x = self._lowpass_baseband(x)

        inst_freq = self._instantaneous_frequency(x)
        inst_freq = self._robust_smooth_frequency(inst_freq, x)

        if self.remove_dc_offset:
            inst_freq = inst_freq - np.median(inst_freq)

        # Map the LoRa baseband frequency excursion into an audible band.
        mapped_freq = self._map_to_audio_band(inst_freq)

        # Time-stretch for audibility.
        stretched_freq = self._time_stretch_trace(mapped_freq)

        # Synthesize a real waveform from the mapped instantaneous frequency.
        audio = self._synthesize_from_frequency(stretched_freq)

        # Final polish.
        audio = self._apply_fade(audio)
        audio = self._final_normalize(audio)

        if self.soft_clip:
            audio = np.tanh(self.soft_clip_drive * audio)
            audio = self._final_normalize(audio)

        if return_debug:
            self.last_debug = {
                "input_samples": len(x),
                "inst_freq_hz": inst_freq,
                "mapped_freq_hz": mapped_freq,
                "stretched_freq_hz": stretched_freq,
                "audio": audio,
            }
        else:
            self.last_debug = {}

        return audio.astype(np.float32, copy=False)

    def save_wav(self, path: str | Path, audio: np.ndarray) -> None:
        """Save a waveform to a 16-bit PCM WAV file."""
        path = Path(path)
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            raise ValueError("audio is empty")
        y = np.clip(y, -1.0, 1.0)
        pcm16 = np.int16(np.round(y * 32767.0))
        wavfile.write(str(path), self.audio_fs, pcm16)

    # ---------------------------------------------------------------------
    # Validation / normalisation
    # ---------------------------------------------------------------------
    def _validate_iq(self, iq: np.ndarray) -> np.ndarray:
        x = np.asarray(iq)
        if x.ndim != 1:
            raise ValueError("iq must be a 1-D array")
        if not np.iscomplexobj(x):
            raise ValueError("iq must be complex-valued")
        x = np.nan_to_num(x.astype(np.complex128, copy=False))
        if x.size < 4:
            raise ValueError("iq is too short")
        return x

    def _normalize_complex_amplitude(self, x: np.ndarray) -> np.ndarray:
        """Normalize complex amplitude to a sane range without destroying phase."""
        mag = np.abs(x)
        peak = float(np.max(mag))
        if peak <= self.amplitude_floor:
            return x.copy()
        return x / peak

    # ---------------------------------------------------------------------
    # Baseband cleanup
    # ---------------------------------------------------------------------
    def _lowpass_baseband(self, x: np.ndarray) -> np.ndarray:
        """Zero-phase low-pass around the LoRa occupied bandwidth."""
        # Keep a bit of guard band above the nominal 125 kHz.
        cutoff_hz = 0.5 * self.lora_bw + self.lowpass_margin_hz
        nyq = 0.5 * self.input_fs
        cutoff_hz = min(cutoff_hz, 0.95 * nyq)
        if cutoff_hz <= 0:
            return x

        wn = cutoff_hz / nyq
        sos = butter(self.lowpass_order, wn, btype="lowpass", output="sos")
        return sosfiltfilt(sos, x)

    # ---------------------------------------------------------------------
    # Instantaneous frequency estimation
    # ---------------------------------------------------------------------
    def _instantaneous_frequency(self, x: np.ndarray) -> np.ndarray:
        """Estimate instantaneous frequency from the phase increment."""
        # phase_inc[n] ≈ arg(x[n] * conj(x[n-1]))
        z = x[1:] * np.conj(x[:-1])
        phase_inc = np.angle(z)
        freq = phase_inc * (self.input_fs / (2.0 * np.pi))

        # Pad one sample so the output length matches the input length.
        if freq.size == 0:
            return np.zeros_like(np.real(x))
        freq = np.concatenate(([freq[0]], freq))
        return freq.astype(np.float64, copy=False)

    def _robust_smooth_frequency(self, freq: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Amplitude-weighted smoothing of the instantaneous frequency."""
        freq = np.asarray(freq, dtype=np.float64)
        amp = np.abs(x).astype(np.float64)
        amp2 = np.maximum(amp * amp, self.amplitude_floor)

        win = int(round(self.smooth_ms * 1e-3 * self.input_fs))
        win = max(3, win)
        if win % 2 == 0:
            win += 1

        if win >= freq.size:
            win = freq.size - 1 if (freq.size - 1) % 2 == 1 else freq.size - 2
            win = max(3, win)

        kernel = np.ones(win, dtype=np.float64)
        numerator = np.convolve(freq * amp2, kernel, mode="same")
        denominator = np.convolve(amp2, kernel, mode="same") + self.amplitude_floor
        smoothed = numerator / denominator

        if self.use_savgol and smoothed.size >= 7:
            # Savitzky-Golay cleans the residual wiggles while preserving chirp edges.
            poly = min(self.savgol_polyorder, 4)
            sg_win = min(win, smoothed.size)
            if sg_win % 2 == 0:
                sg_win -= 1
            if sg_win >= 5:
                smoothed = savgol_filter(smoothed, window_length=sg_win, polyorder=poly, mode="interp")

        # Clip extreme spikes caused by low-SNR samples.
        lo = np.percentile(smoothed, 100.0 - self.robust_percentile)
        hi = np.percentile(smoothed, self.robust_percentile)
        smoothed = np.clip(smoothed, lo, hi)
        return smoothed

    # ---------------------------------------------------------------------
    # Audible mapping
    # ---------------------------------------------------------------------
    def _map_to_audio_band(self, freq: np.ndarray) -> np.ndarray:
        """Compress the RF-band frequency excursion into an audible band."""
        freq = np.asarray(freq, dtype=np.float64)

        # We intentionally do not preserve the original 125 kHz absolute scale.
        # Instead, we map the full LoRa excursion into a compact audible window.
        #
        # Example default:
        #   input excursion ~ ±62.5 kHz
        #   output excursion ~ ±4 kHz around 6 kHz -> roughly 2 kHz..10 kHz
        src_half_span = 0.5 * self.lora_bw
        dst_half_span = self.audio_span_hz
        scale = dst_half_span / max(src_half_span, self.amplitude_floor)

        mapped = self.audio_center_hz + scale * freq

        # Keep a safety margin below Nyquist in case of occasional excursions.
        nyq = 0.5 * self.audio_fs
        lower = 50.0
        upper = 0.92 * nyq
        mapped = np.clip(mapped, lower, upper)
        return mapped

    def _time_stretch_trace(self, trace: np.ndarray) -> np.ndarray:
        """Stretch the time axis by interpolation."""
        trace = np.asarray(trace, dtype=np.float64)
        if trace.size < 2 or self.time_stretch == 1.0:
            return trace.copy()

        in_dur = trace.size / self.input_fs
        out_dur = in_dur * self.time_stretch
        n_out = max(2, int(round(out_dur * self.audio_fs)))

        src_t = np.arange(trace.size, dtype=np.float64) / self.input_fs
        # Each output time corresponds to an earlier input time because we slow it down.
        dst_src_t = (np.arange(n_out, dtype=np.float64) / self.audio_fs) / self.time_stretch
        stretched = np.interp(dst_src_t, src_t, trace, left=trace[0], right=trace[-1])
        return stretched

    # ---------------------------------------------------------------------
    # Audio synthesis
    # ---------------------------------------------------------------------
    def _synthesize_from_frequency(self, freq_hz: np.ndarray) -> np.ndarray:
        """Integrate the target instantaneous frequency into a real waveform."""
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        if freq_hz.size == 0:
            return np.zeros(0, dtype=np.float64)

        phase = np.cumsum(2.0 * np.pi * freq_hz / self.audio_fs)
        y = np.cos(phase)
        return y

    def _apply_fade(self, audio: np.ndarray) -> np.ndarray:
        """Apply a short fade in/out to suppress clicks."""
        y = np.asarray(audio, dtype=np.float64).copy()
        n = y.size
        if n == 0:
            return y

        fade_len = int(round(self.fade_ms * 1e-3 * self.audio_fs))
        fade_len = max(0, min(fade_len, n // 2))
        if fade_len < 2:
            return y

        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade_len, dtype=np.float64)) ** 2
        y[:fade_len] *= ramp
        y[-fade_len:] *= ramp[::-1]
        return y

    def _final_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Peak-normalize the output to [-1, 1]."""
        y = np.asarray(audio, dtype=np.float64).copy()
        y *= float(self.output_gain)
        peak = float(np.max(np.abs(y)))
        if peak <= self.amplitude_floor:
            return y
        y = y / peak
        return y

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def process_and_save(self, iq: np.ndarray, path: str | Path) -> np.ndarray:
        """Process IQ and directly save the WAV file."""
        audio = self.process(iq)
        self.save_wav(path, audio)
        return audio

    def inspect(self, iq: np.ndarray) -> Dict[str, Any]:
        """Run the chain and expose intermediate arrays for debugging."""
        _ = self.process(iq, return_debug=True)
        return self.last_debug.copy()


# -------------------------------------------------------------------------
# Optional standalone helper
# -------------------------------------------------------------------------
def sonify_lora_iq(
    iq: np.ndarray,
    input_fs: float = 2_400_000.0,
    audio_fs: int = 48_000,
    time_stretch: float = 3.0,
) -> np.ndarray:
    """Convenience function for quick experiments."""
    dsp = Dsp(input_fs=input_fs, audio_fs=audio_fs, time_stretch=time_stretch)
    return dsp.process(iq)
