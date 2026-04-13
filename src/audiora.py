import time
import rtlsdr.rtlsdrtcp.base

from ppk2_api import PPK2_MP as PPK2_API
from lora_detector import LoraDetector, Acquisition
from sdr_receiver import SdrReceiver, RadioParameters
from dsp import Dsp
from audio_player import AudioPlayer
from utils import *

SDR_READ_SIZE = rtlsdr.rtlsdrtcp.base.DEFAULT_READ_SIZE

class Audiora:

    def __init__(self):

        # Radio receiver parameters
        self.rfDevice = SdrReceiver()
        self.loraDetector = LoraDetector(RadioParameters.samplerate)
        self.maxRecTime = 10  # seconds
        self.sr = 2_400_000
        self.bw = 125_000
        self.fc = 868_300_000

        # PPK2
        self.ppk2 = None

        # Signal processing module
        self.dsp = Dsp()

        # Audio module
        self.audio_sr: int = 48000
        self.audioDeviceName = 'Minifuse'
        self.audioPlayer = AudioPlayer(samplerate=self.audio_sr, device_name=self.audioDeviceName, channels=2)

    def init(self):

        self.rfDevice.configureSdr(
            sr = self.sr,
            bw = RadioParameters.bandwidth,
            fc = RadioParameters.centralfrequency,
             g = RadioParameters.gain
        )
        print("RTL-SDR-RECEIVER Initialized")

        self.loraDetector.threshold_calibration(self.rfDevice.read(SDR_READ_SIZE * 2048))
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

    def start(self, useCsv):

        # Pipeline:
        # Power DUT > listen > detect > record > process > output as .wav

        if not useCsv:
            print("PPK2: powering on DUT.")
            self.ppk2.toggle_DUT_power("ON")
            print("Listening...")
            iq_chunk = Acquisition.listen(self.rfDevice, self.loraDetector) # blocking
            print("LoRa modulation detected. Starting recording...")
            iq_recording = Acquisition.record(iq_chunk,self.sr,self.maxRecTime,self.rfDevice,self.loraDetector)
            print("Recording done.")
            time.sleep(2)
            self.ppk2.toggle_DUT_power("OFF")
            print("PPK2: DUT powered off.")
            show_spectrogram(iq_recording, "spectrogram of signal out of the radio recording")
        else:
            print("Using CSV instead.")
            iq_recording = np.array(load_from_csv("../misc/radio_output_iq.csv"))

        print("Processing signal...")
        audio = self.dsp.process(iq_recording)

        print("Processing done. Exporting audio file ...")
        self.audioPlayer.save_wav(audio, "audibleLoRa.wav")
        self.audioPlayer.play(audio)
        print("End of program.")


if __name__ == '__main__':
    audiora = Audiora()
    audiora.init()
    audiora.start(useCsv=True)
    audiora.audioPlayer.stop()
    exit()

