import numpy as np
from rtlsdr import *

"""
    RTL-SDR BLog v4 datasheet: https://www.rtl-sdr.com/wp-content/uploads/2024/12/RTLSDR_V4_Datasheet_V_1_0.pdf
"""


"""
    LoRaWAN regional parameters specification:
    https://resources.lora-alliance.org/technical-specifications/rp002-1-0-4-regional-parameters
    From page 32 - table 7: EU863-870 Join-Request channel list
    There are only 3 possible channels: 868.1, 868.3 and 868.5 MHz with a bandwidth of 125 kHz, at DR0 to DR5

    => let's use bandwidth = 125kHz, and centerfrequency = 868.3MHz and hope for the device to Join on this channel.
"""
class RadioParameters:
    samplerate = 2.4e6
    bandwidth = 125e3
    centralfrequency = 868.3e6
    gain = 'auto'

class SdrReceiver:

    def __init__(self):
        self.sdr = RtlSdr()

    def configureSdr(self, sr: float, bw: float, fc: float, g):
        self.sdr.sample_rate = sr
        self.sdr.bandwidth = bw
        self.sdr.center_freq = fc
        self.sdr.gain = g

    def read(self, read_size: int) -> np.ndarray:
        return self.sdr.read_samples(read_size)

    def close(self):
        self.sdr.close()