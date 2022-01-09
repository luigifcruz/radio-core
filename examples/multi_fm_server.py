import sys
import zmq
import numpy as np
import sounddevice as sd
from threading import Thread
from dataclasses import dataclass

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, RingBuffer, FM, MFM, WBFM, Tuner


@dataclass
class Channel:
    frequency: float    # The FM station frequency.
    bandwidth: float    # The FM station bandwidth. (240-256 kHz).
    audio_fs: float     # The audio sample-rate.
    demodulator: FM     # The demodulator to use (WBFM, MFM, or FM).


@dataclass
class Config:
    enable_cuda: bool = False       # If True, enable CUDA demodulation.
    input_rate: float = 10e6        # The SDR RX bandwidth.
    device_name: str = "airspy"     # The SoapySDR device string.
    deemphasis: float = 75e-6       # 75e-6 for Americas and Korea, otherwise 50e-6.
    channels = [
        Channel(96.9e6, 240e3, 48e3, WBFM),
        Channel(94.5e6, 240e3, 48e3, MFM),
        Channel(97.5e6, 240e3, 48e3, FM),
    ]


class SdrDevice(Thread):

    def __init__(self, config: Config, tuner: Tuner):
        super().__init__()
        self.config = config
        self.tuner = tuner

        print("Configuring SDR device...")
        self.sdr = Device({"driver": self.config.device_name})
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.tuner.input_bandwidth)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.tuner.input_frequency)
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, True)
        self.rx = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

        print("Allocating SDR device buffers...")
        self.buffer = RingBuffer(self.config.input_rate * 3,
                                 cuda=self.config.enable_cuda)

    @property
    def output(self) -> RingBuffer:
        return self.buffer

    def run(self):
        tbuf = Buffer(2**16, cuda=self.config.enable_cuda)
        self.sdr.activateStream(self.rx)
        self.running = True

        while self.running:
            c = self.sdr.readStream(self.rx, [tbuf.data], tbuf.size, timeoutUs=2**37)
            self.buffer.append(tbuf.data[:c.ret])

    def stop(self):
        self.sdr.deactivateStream(self.rx)
        self.sdr.closeStream(self.rx)
        self.running = False
        self.join()


class Dsp(Thread):

    def __init__(self, config: Config, tuner: Tuner, socket, data_in: RingBuffer):
        super().__init__()
        self.tuner = tuner
        self.config = config
        self.socket = socket
        self.data_in = data_in

    def run(self):
        tbuf = Buffer(self.config.input_rate, cuda=self.config.enable_cuda)
        self.running = True

        while self.running:
            occupancy = (self.data_in.occupancy / self.data_in.capacity) * 100
            print(f"DSP buffer occupancy: {occupancy:.2f}%")

            if not self.data_in.popleft(tbuf.data):
                continue

            self.tuner.load(tbuf.data)

            for channel in self.tuner.channels():
                tmp = self.tuner.run(channel.index)
                tmp = channel.demodulator.run(tmp)
                tmp = tmp.tobytes()

                payload = [channel.address_bytes, tmp]
                self.socket.send_multipart(payload)

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

    for channel in config.channels:
        # Initialize a demodulator for each channel.
        demod = channel.demodulator(channel.bandwidth,
                                    channel.audio_fs,
                                    deemphasis=config.deemphasis,
                                    cuda=config.enable_cuda)

        # Commit channel configuration to Tuner.
        tuner.add_channel(channel.frequency, channel.bandwidth, demod)

    # We request a bandwidth since Airspy doesn't support variable fs.
    tuner.request_bandwidth(config.input_rate)

    # Configure SDR device thread.
    rx = SdrDevice(config, tuner)
    dsp = Dsp(config, tuner, socket, rx.output)

    try:
        print(f"Starting processing {len(config.channels)} radios...")
        rx.start()
        dsp.start()
        rx.join()
    except KeyboardInterrupt:
        dsp.stop()
        rx.stop()
        sys.exit('\nInterrupted by user. Closing...')
