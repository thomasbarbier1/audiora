import time
import numpy as np
from scipy.io import loadmat
import rtlsdr.rtlsdrtcp.base
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from lora_detector import LoraDetector
from sdr_receiver import SdrReceiver, RadioParameters
from acquisition import Acquisition
from signal_processor import SignalProcessor
from audio_player import AudioPlayer

SDR_READ_SIZE = rtlsdr.rtlsdrtcp.base.DEFAULT_READ_SIZE

class Audiora:

    def __init__(self):

        # Radio receiver parameters
        self.rfDevice = SdrReceiver()

        self.sr = RadioParameters.samplerate
        self.bw = RadioParameters.bandwidth
        self.fc = RadioParameters.centralfrequency
        # self.sr = 2.048e6 # modified for the test
        # self.bw = 125e3 # modified for the test
        # self.fc = 867.4e6 # modified for the test

        # Sub modules
        self.maxRecTime = 5 # seconds
        self.loraDetector = LoraDetector(RadioParameters.samplerate)
        self.dsp = SignalProcessor(self.sr, RadioParameters.bandwidth, stretching_factor=4)
        self.audioPlayer =AudioPlayer(samplerate=44_100.0, device_name="MiniFuse", channels=2)

    def init(self):

        self.rfDevice.configureSdr(
            sr = self.sr,
            bw = RadioParameters.bandwidth,
            fc = RadioParameters.centralfrequency,
             g = RadioParameters.gain
        )
        print("RTL-SDR-RECEIVER Initialized")

        self.detector_calibration()
        print("Detection threshold calibrated.")

    def detector_calibration(self):
        self.loraDetector.threshold_calibration(self.rfDevice.read(SDR_READ_SIZE * 2048))

    def show_chirps(self, iq):
        f, t, Sxx = spectrogram(iq, fs=1.0, return_onesided=False)
        plt.imshow(10 * np.log10(np.abs(Sxx)), aspect='auto', origin='lower')
        plt.title("Spectrogram")
        plt.show()

    def test(self):

        """
        This test uses a free-online bank of IQ recording.
        Used to test the dsp module without the radio module.
        """

        # from .mat file
        print("Loading .mat file")
        data = loadmat("lorasf12_g0.0dB_att24dB_freq867.4MHz_0.mat")
        iq_raw = data["IQ_samples"]
        iq_samples = iq_raw.squeeze().astype(np.complex64)
        iq_recording = iq_samples / np.max(np.abs(iq_samples))

        # Visualize of the lora modulation (enjoy!!)
        self.show_chirps(iq_recording)

        # process
        print("Processing lora to audio")
        # offset mesuré sur le spectrogramme : indice 85 sur 256 bins
        # mais l'axe est centré (fftshift), donc offset réel :
        if_offset = (85 - 128) / 256 * self.sr  # ≈ -334 kHz
        sound = self.dsp.compute(iq_recording, if_offset)

        # play
        print("Playing sound")
        self.audioPlayer.play(sound)

    def start(self):
        while True:

            # listen
            iq_chunk = Acquisition.listen(self.rfDevice, self.loraDetector)

            # record
            iq_recording = Acquisition.record(iq_chunk,
                                              self.sr,
                                              self.maxRecTime,
                                              self.rfDevice,
                                              self.loraDetector)

            # Visualize of the lora modulation (enjoy!!)
            self.show_chirps(iq_recording)

            # process
            sound = self.dsp.compute(iq_recording)

            # play
            self.audioPlayer.play(sound)

    @staticmethod
    def show_psd(samples, fs, fc):
        # use matplotlib to estimate and plot the PSD
        plt.psd(samples, Fs=fs, Fc=fc)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Relative power (dB)')
        plt.show()

if __name__ == '__main__':
    audiora = Audiora()
    audiora.init()
    audiora.start()
    # audiora.test()
    audiora.audioPlayer.stop()
    print("Done")
    exit()

