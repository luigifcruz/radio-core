import sys
import time
import queue
import sounddevice as sd
from threading import Thread
from dataclasses import dataclass

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, RingBuffer, FM, MFM, WBFM, Decimate


@dataclass
class Config:
    enable_cuda: bool = False       # If True, enable CUDA demodulation.
    frequency: float = 96.9e6       # Set the FM station frequency.
    deemphasis: float = 75e-6       # 75e-6 for Americas and Korea, otherwise 50e-6.
    input_rate: float = 10e6        # SDR RX bandwidth.
    demod_rate: float = 250e3       # FM station bandwidth. (240-256 kHz).
    audio_rate: float = 48e3        # Audio bandwidth (32-48 kHz).
    device_name: str = "airspy"     # SoapySDR device string.
    demodulator = WBFM              # Demodulator (WBFM, MFM, or FM).


class SdrDevice(Thread):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.running = False

        print("Configuring SDR device...")
        self.sdr = Device({"driver": self.config.device_name})
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.config.input_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.config.frequency)
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, True)
        self.rx = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

        print("Allocating SDR device buffers...")
        self.buffer = RingBuffer(self.config.input_rate * 3,
                                 cuda=self.config.enable_cuda)

    @property
    def output(self) -> RingBuffer:
        return self.buffer

    def run(self):
        tmp_buffer = Buffer(2**16, cuda=self.config.enable_cuda)

        self.sdr.activateStream(self.rx)
        self.running = True

        while self.running:
            c = self.sdr.readStream(self.rx,
                                    [tmp_buffer.data],
                                    tmp_buffer.size,
                                    timeoutUs=500000)
            if c.ret > 0:
                self.buffer.put(tmp_buffer.data[:c.ret])

    def stop(self):
        self.sdr.deactivateStream(self.rx)
        self.sdr.closeStream(self.rx)
        self.running = False
        self.join()


class Dsp(Thread):

    def __init__(self, config: Config, data_in: RingBuffer):
        super().__init__()
        self.config = config
        self.data_in = data_in
        self.running = False

        print("Configuring DSP...")
        self.demod = self.config.demodulator(self.config.demod_rate,
                                             self.config.audio_rate,
                                             deemphasis=self.config.deemphasis,
                                             cuda=self.config.enable_cuda)
        self.decim = Decimate(self.config.input_rate,
                              self.config.demod_rate,
                              cuda=self.config.enable_cuda)

        print("Allocating DSP buffers...")
        self.que = queue.Queue()

    @property
    def output(self) -> queue.Queue:
        return self.que

    def run(self):
        tmp_buffer = Buffer(self.config.input_rate, cuda=self.config.enable_cuda)

        self.running = True

        while self.running:
            if not self.data_in.get(tmp_buffer.data):
                continue

            tmp = self.decim.run(tmp_buffer.data)
            tmp = self.demod.run(tmp)

            self.que.put_nowait(tmp)

    def stop(self):
        self.running = False
        self.join()


if __name__ == "__main__":
    config = Config()

    # Get frequency from command-line if available.
    if len(sys.argv) > 1:
        config.frequency = float(sys.argv[1])

    # Configure SDR device thread.
    rx = SdrDevice(config)
    dsp = Dsp(config, rx.output)

    # Define demodulation callback. This should not block.
    def process(outdata, *_):
        if not dsp.output.empty():
            outdata[:] = dsp.output.get_nowait()
        else:
            outdata[:] = 0.0

    # Configure sound device stream.
    stream = sd.OutputStream(blocksize=int(config.audio_rate),
                             callback=process,
                             samplerate=int(config.audio_rate),
                             channels=dsp.demod.channels)

    try:
        print("Starting playback...")
        rx.start()
        dsp.start()
        stream.start()

        # Busy loop until interrupted by KeyboardInterrupt.
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        dsp.stop()
        rx.stop()
        sys.exit('\nInterrupted by user. Closing...')
