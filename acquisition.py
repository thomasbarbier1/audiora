import numpy as np
from lora_detector import LoraDetector
from sdr_receiver import SdrReceiver

class Acquisition:

    @staticmethod
    def record(smp: np.ndarray, sr: float, maxRecTime_s: int, rfDevice: SdrReceiver, loraDetector: LoraDetector):
        _buffer = []
        while smp is not None and len(_buffer) < (sr * maxRecTime_s):
            _buffer.append(smp)
            IQ = rfDevice.read(1024)
            smp = loraDetector.feed(IQ)
        return np.concatenate(_buffer)

    @staticmethod
    def listen(rfDevice: SdrReceiver, loraDetector: LoraDetector):
        smp = None
        while smp is None:
            IQ = rfDevice.read(1024)
            smp = loraDetector.feed(IQ)
        return smp