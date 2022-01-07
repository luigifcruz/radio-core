import sys

import numpy as np
import sounddevice as sd
from threading import Thread
from dataclasses import dataclass

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, RingBuffer, FM, MFM, WBFM, Decimate


@dataclass
class Config:
    enable_cuda: bool = False       # If True, enable CUDA demodulation.
    frequency: float = 96.9e6       # Set the FM station frequency.
    deemphasis: float = 75e-6       # 50e-6 for World and 75e-6 for Americas and Korea.
    input_rate: float = 10e6        # SDR RX bandwidth.
    demod_rate: float = 250e3       # FM station bandwidth. (240-256 kHz).
    audio_rate: float = 48e3        # Audio bandwidth (32-48 kHz).
    device_rate: int = 1024         # Device buffer size.
    device_name: str = "airspy"     # SoapySDR device string.
    demodulator = WBFM              # Demodulator (WBFM, MFM, or FM).


class SdrDevice(Thread):

    def __init__(self, config: Config, ring_buffer: RingBuffer):
        super().__init__()
        self.config = config
        self.ring_buffer = ring_buffer

        print("Configuring SDR device...")
        self.sdr = Device({"driver": self.config.device_name})
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, True)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.config.input_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.config.frequency)
        self.rx = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

        print("Allocating SDR device buffers...")
        self.buffer = Buffer(self.config.device_rate, dtype=np.complex64,
                             cuda=self.config.enable_cuda)

    def run(self):
        self.running = True
        self.sdr.activateStream(self.rx)

        while self.running:
            self.sdr.readStream(self.rx, [self.buffer.data],
                                self.config.device_rate, timeoutUs=2**37)
            self.ring_buffer.append(self.buffer.data)

    def stop(self):
        self.sdr.deactivateStream(self.rx)
        self.sdr.closeStream(self.rx)
        self.running = False

# Define demodulation callback. This should not block.
def process(outdata, *_):
    print(ring_buffer.occupancy)
    if not ring_buffer.popleft(output_buffer.data):
        return

    outdata[:] = demod.run(decim.run(output_buffer.data))

config = Config()

# Start DSP processors.
print("Configuring DSP...")
demod = config.demodulator(config.demod_rate, config.audio_rate, cuda=config.enable_cuda)
decim = Decimate(config.input_rate, config.demod_rate, cuda=config.enable_cuda)

# Allocate buffers.
print("Allocating buffers...")
ring_buffer = RingBuffer(config.input_rate * 10, dtype=np.complex64, cuda=config.enable_cuda)
output_buffer = Buffer(config.input_rate, dtype=np.complex64, cuda=config.enable_cuda)

# Start collecting data.
print("Starting device and audio stream...")
stream = sd.OutputStream(blocksize=int(config.audio_rate), callback=process,
                         samplerate=int(config.audio_rate), channels=demod.channels)

rx = SdrDevice(config, ring_buffer)
rx.start()
stream.start()

print("join")
rx.join()
print("outjoin")

try:
    print("Starting playback...")
except KeyboardInterrupt:
    sys.exit('\nInterrupted by user. Closing...')
