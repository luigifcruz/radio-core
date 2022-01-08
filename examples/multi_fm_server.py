import sys
import zmq
import numpy as np
import sounddevice as sd
from threading import Thread
from dataclasses import dataclass

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, RingBuffer, FM, MFM, WBFM, Tuner


@dataclass
class Config:
    enable_cuda: bool = False       # If True, enable CUDA demodulation.
    frequency: float = 0.0          # The frequency will be set by the Tuner.
    deemphasis: float = 75e-6       # 50e-6 for World and 75e-6 for Americas and Korea.
    input_rate: float = 10e6        # SDR RX bandwidth.
    demod_rate: float = 250e3       # FM station bandwidth. (240-256 kHz).
    audio_rate: float = 48e3        # Audio bandwidth (32-48 kHz).
    device_rate: int = 1024         # Device buffer size.
    device_name: str = "airspy"     # SoapySDR device string.
    demodulator = WBFM              # Demodulator (WBFM, MFM, or FM).


class SdrDevice(Thread):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        print("Configuring SDR device...")
        self.sdr = Device({"driver": self.config.device_name})
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, True)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.config.input_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.config.frequency)
        self.rx = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

        print("Allocating SDR device buffers...")
        self.tmp_buffer = Buffer(self.config.device_rate, dtype=np.complex64,
                                 cuda=self.config.enable_cuda)
        self.ring_buffer = RingBuffer(self.config.input_rate * 10,
                                      dtype=np.complex64)

    @property
    def output(self) -> RingBuffer:
        return self.ring_buffer

    def run(self):
        self.running = True
        self.sdr.activateStream(self.rx)

        while self.running:
            self.sdr.readStream(self.rx, [self.tmp_buffer.data],
                                self.config.device_rate, timeoutUs=2**37)
            self.ring_buffer.append(self.tmp_buffer.data)

    def stop(self):
        self.sdr.deactivateStream(self.rx)
        self.sdr.closeStream(self.rx)
        self.running = False
        self.join()


class Dsp(Thread):

    def __init__(self, config: Config, tuner: Tuner, socket, input: RingBuffer):
        super().__init__()
        self.config = config
        self.input = input
        self.tuner = tuner
        self.socket = socket

        print("Configuring DSP...")
        self.demod = []
        self.address = []

        for _, c in enumerate(self.tuner.channels):
            _demod = self.config.demodulator(self.config.demod_rate,
                                             self.config.audio_rate,
                                             cuda=self.config.enable_cuda)
            self.demod.append(_demod)

            _address = int(c.center_frequency).to_bytes(4, byteorder='little')
            self.address.append(_address)

        print("Allocating DSP buffers...")
        self.tmp_buffer = Buffer(self.config.input_rate, dtype=np.complex64,
                                 cuda=self.config.enable_cuda)

    def run(self):
        self.running = True

        while self.running:
            if not self.input.popleft(self.tmp_buffer.data):
                continue

            tmp = self.tuner.load(self.tmp_buffer.data)

            for i, _ in enumerate(self.tuner.channels):
                tmp = self.tuner.run(i)
                tmp = self.demod[i].run(tmp)
                tmp = tmp.tobytes()

                self.socket.send_multipart([self.address[i], tmp])

    def stop(self):
        self.running = False
        self.join()


if __name__ == "__main__":
    config = Config()

    # Configure ZeroMQ server.
    context = zmq.Context()
    context.setsockopt(zmq.IPV6, True)
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    # Configure Tuner.
    tuner = Tuner(cuda=config.enable_cuda)
    tuner.add_channel(94.5e6, config.demod_rate)
    tuner.add_channel(97.5e6, config.demod_rate)
    tuner.add_channel(96.9e6, config.demod_rate)
    tuner.request_bandwidth(config.input_rate)
    config.frequency = tuner.input_frequency

    # Configure SDR device thread.
    rx = SdrDevice(config)
    dsp = Dsp(config, tuner, socket, rx.output)

    try:
        print("Starting processing...")
        rx.start()
        dsp.start()
        rx.join()
    except KeyboardInterrupt:
        dsp.stop()
        rx.stop()
        sys.exit('\nInterrupted by user. Closing...')
