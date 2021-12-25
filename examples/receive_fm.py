import sys

import numpy as np
import sounddevice as sd

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, Carrousel, Chopper, MFM, WBFM, Resample


def receive(queue):
    # Load, fill, and enqueue buffer.
    with queue.enqueue() as buffer:
        for chunk in chopr.chop(buffer):
            sdr.readStream(rx, [chunk], len(chunk), timeoutUs=int(1e9))

    # Start audio when buffer reaches N samples.
    if queue.occupancy >= 4 and not stream.active:
        stream.start()


def process(outdata, *_):
    if not queue.is_healthy:
        return

    # Load, demod, and play buffer.
    with queue.dequeue() as buffer:
        demoded = demod.run(decim.run(buffer))

        if demod.channels == 2:
            outdata[:] = np.dstack(demoded)
        else:
            outdata[:, 0] = demoded


if __name__ == "__main__":
    enable_cuda: bool = False       # If True, enable CUDA demodulation.
    frequency: float = 96.9e6       # Set the FM station frequency.
    deemphasis: float = 75e-6       # 50e-6 for World and 75e-6 for Americas and SKorea.
    input_rate: float = 2.5e6        # SDR RX bandwidth.
    radio_rate: float = 240e3       # FM station bandwidth. (240-256 kHz).
    output_rate: float = 48e3       # Audio bandwidth (32-48 kHz).
    device_buffer: float = 2000     # SDR device buffer size.
    device_name: str = "airspy"     # SoapySDR device string.
    demodulator = MFM              # Demodulator (WBFM, MFM, or FM).

    # SoapySDR configuration.
    print("Configuring device...")
    sdr = Device({"driver": device_name})
    sdr.setGainMode(SOAPY_SDR_RX, 0, True)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, input_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, frequency)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    # Queue and shared memory allocation.
    print("Configuring DSP...")
    chopr = Chopper(input_rate, device_buffer)
    queue = Carrousel([Buffer(input_rate, cuda=enable_cuda) for _ in range(10)])
    decim = Resample(input_rate, radio_rate, cuda=enable_cuda)
    demod = demodulator(radio_rate, output_rate, cuda=enable_cuda)

    # Start collecting data.
    print("Starting device and audio stream...")
    sdr.activateStream(rx)
    stream = sd.OutputStream(blocksize=int(output_rate), callback=process,
                             samplerate=int(output_rate), channels=demod.channels)

    try:
        print("Starting playback...")
        while True:
            receive(queue)
    except KeyboardInterrupt:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        sys.exit('\nInterrupted by user. Closing...')
