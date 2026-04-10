import numpy as np
import logging

class LoraDetector:
    def __init__(self, sr, buffer_size=256, calibration_factor=3.0):
        self.threshold = None
        self.calibration_factor = calibration_factor
        self.buffer_size = buffer_size
        self._buffer = [] # np.ndarray(self.buffer_size, dtype=np.complex64)

    def threshold_calibration(self, samples: np.ndarray) -> None:
        noise_floor = np.mean(np.abs(samples))
        self.threshold = self.calibration_factor * noise_floor
        logging.info(f"Calibration : noise_floor={noise_floor:.4f}, threshold={self.threshold:.4f}")

    def feed(self, samples: np.ndarray) -> np.ndarray | None:
        self._buffer.append(samples)
        if len(self._buffer) < self.buffer_size:
            return None

        window = np.concatenate(self._buffer)
        self._buffer.clear()

        amplitude = np.mean(np.abs(window))
        if amplitude > self.threshold:
            return window

        return None
