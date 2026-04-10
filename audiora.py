import time
import numpy as np
from scipy.io import loadmat
import rtlsdr.rtlsdrtcp.base

from ppk2_api import PPK2_MP as PPK2_API
from lora_detector import LoraDetector, Acquisition
from sdr_receiver import SdrReceiver, RadioParameters
from signal_processor import SignalProcessor, downsampler
from audio_player import AudioPlayer
from utils import *

SDR_READ_SIZE = rtlsdr.rtlsdrtcp.base.DEFAULT_READ_SIZE

class Audiora:

    def __init__(self):

        # test duration
        self.duration = 60

        # Radio receiver parameters
        self.rfDevice = SdrReceiver()
        self.loraDetector = LoraDetector(RadioParameters.samplerate)
        self.maxRecTime = 5  # seconds
        self.sr = RadioParameters.samplerate
        self.bw = RadioParameters.bandwidth
        self.fc = RadioParameters.centralfrequency
        # self.sr = 2.048e6 # modified for the test
        # self.bw = 125e3 # modified for the test
        # self.fc = 867.4e6 # modified for the test

        # PPK2
        self.ppk2 = None

        # Signal processing module
        self.dsp = SignalProcessor(self.sr, RadioParameters.bandwidth, stretching_factor=4)

        # Audio module
        self.audio_sr: int = 44100
        self.audioDeviceName = 'Jabra'
        self.audioPlayer =AudioPlayer(samplerate=self.audio_sr, device_name=self.audioDeviceName, channels=2)

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

        ppk2s_connected = PPK2_API.list_devices()
        if len(ppk2s_connected) == 1:
            ppk2_port = ppk2s_connected[0][0]
            print(f"Found PPK2.")
        else:
            raise ValueError("Zero or more than one PPK2 connected!\n")
        self.ppk2 = PPK2_API(ppk2_port,buffer_max_size_seconds=1,buffer_chunk_seconds=0.01,
                     timeout=1, write_timeout=1, exclusive=True)
        self.ppk2.get_modifiers()
        self.ppk2.set_source_voltage(3600)
        self.ppk2.use_source_meter()  # set source meter mode
        print("PPK2 connected and ready.")

    def detector_calibration(self):
        self.loraDetector.threshold_calibration(self.rfDevice.read(SDR_READ_SIZE * 2048))

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

    def start(self, useCsv, csvName):

        # timeout = time.monotonic() + self.duration
        # while time.monotonic() < timeout:

        if not useCsv:
            # start ppk2
            self.ppk2.toggle_DUT_power("ON")  # enable DUT power
            print("PPK2: powering on DUT.")

            # listen
            print("Listening...")
            iq_chunk = Acquisition.listen(self.rfDevice, self.loraDetector)

            # record
            print("LoRa modulation detected. Starting recording...")
            iq_recording = Acquisition.record(iq_chunk,
                                              self.sr,
                                              self.maxRecTime,
                                              self.rfDevice,
                                              self.loraDetector)
            # stop ppk2
            self.ppk2.toggle_DUT_power("OFF")
            print("PPK2: DUT powered off.")

            # Show spectrogram + time_domain
            show_radio_output(iq_recording, self.sr, "time signal out of the radio recording")
            show_spectrogram(iq_recording, "spectrogram of signal out of the radio recording")

            export_to_csv(iq_recording, "radio_output_iq.csv")

            # Pre-processing: anti-alias > downsampling
            # We to downsample from 2.4MHz to 125kHz ==> so the downsampling factor = 19.2
            # so 'up' argument = 1 and 'down' argument = 19.2
            # As 1/19.2 = 5/96 we will use up=5 and down=96 instead, in order to use integers
            iq_recording = self.dsp.preProcess(iq_recording)
            iq_recording = downsampler(iq_recording, up=5, down=96)
            print(f"Signal preprocessed: downsampled from {self.sr} MHz to {self.bw} kHz")

            export_to_csv(iq_recording, "Pre-processed_iq.csv")

        else:
            iq_recording = load_from_csv(csvName)

        # Show again spectrogram + time_domain
        show_radio_output(iq_recording, self.bw, "time signal after pre-processing")
        show_spectrogram(iq_recording, "spectrogram after pre-processing")

        # process
        print("Recording done. Processing signal...")
        sound = self.dsp.compute(iq_recording)
        print("Processing done.")

        # save as .wav and play
        show_audioSignal(sound, self.audio_sr)
        self.audioPlayer.save_wav(sound)
        self.audioPlayer.play(sound)


if __name__ == '__main__':
    audiora = Audiora()
    audiora.init()
    audiora.start(useCsv=True, csvName="Pre-processed_iq.csv")
    # audiora.test()
    audiora.audioPlayer.stop()
    exit()

