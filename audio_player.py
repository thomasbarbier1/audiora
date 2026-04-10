import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self, samplerate=48000, device_name="MiniFuse", channels=2):
        self.samplerate = samplerate
        self.channels = channels
        self.device = self._find_device(device_name)

        if self.device is None:
            raise RuntimeError(f"Audio device containing '{device_name}' not found")

    def _find_device(self, name_substring):
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if name_substring.lower() in dev['name'].lower():
                # On privilégie un device de sortie
                if dev['max_output_channels'] > 0:
                    return i
        return None

    def play(self, signal, normalize=True):
        """
        signal: ndarray 1D (mono) ou 2D (samples, channels)
        """

        if signal.ndim == 1:
            # mono → stéréo si nécessaire
            if self.channels == 2:
                signal = np.column_stack((signal, signal))
        elif signal.ndim == 2:
            pass
        else:
            raise ValueError("Signal must be 1D or 2D numpy array")

        # conversion float32 obligatoire pour PortAudio
        signal = signal.astype(np.float32)

        if normalize:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val * 0.9  # marge

        sd.play(
            signal,
            samplerate=self.samplerate,
            device=self.device,
            blocking=True
        )

    def stop(self):
        sd.stop()