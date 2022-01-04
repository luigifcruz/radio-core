import sys

import numpy as np
import sounddevice as sd

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, Tuner, Carrousel, Chopper, FM, MFM, WBFM, Decimate


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

        for i, _ in enumerate(tuner.channels):
            if demod.channels == 2:
                outdata[:] = np.dstack(demoded)
            else:
                outdata[:, 0] = demoded


if __name__ == "__main__":
    enable_cuda: bool = False       # If True, enable CUDA demodulation.
    deemphasis: float = 75e-6       # 50e-6 for World and 75e-6 for Americas and SKorea.
    input_rate: float = 2.4e6       # SDR RX bandwidth.
    demod_rate: float = 240e3       # FM station bandwidth. (240-256 kHz).
    audio_rate: float = 48e3        # Audio bandwidth (32-48 kHz).
    device_name: str = "sdrplay"    # SoapySDR device string.
    device_buffer: int = 1024       # Device buffer size.
    buffer_multiplier: int = 2000   # Multiplier of device buffer.
    demodulator = WBFM              # Demodulator (WBFM, MFM, or FM).

    print("Buffer Size:")
    print(f"    Device Buffer: {device_buffer}")
    print(f"    Input Buffer: {input_buffer}")
    print(f"    Demod Buffer: {demod_buffer}")
    print(f"    Audio Buffer: {audio_buffer}")

    # Queue and shared memory allocation.
    print("Configuring DSP...")
    tuner = Tuner(cuda=enable_cuda)

    tuner.add_channel(94.5e6, demod_rate)
    tuner.add_channel(88.5e6, demod_rate)
    tuner.add_channel(102.5e6, demod_rate)
    tuner.add_channel(97.5e6, demod_rate)
    tuner.add_channel(96.9e6, demod_rate)

    # Calculate buffer sizes.
    input_buffer: int = device_buffer * buffer_multiplier
    demod_buffer: int = input_buffer / (input_rate / demod_rate)
    audio_buffer: int = demod_buffer / (demod_rate / audio_rate)
    tuner.request_bandwidth(input_rate)

    chopr = Chopper(input_buffer, device_buffer)
    queue = Carrousel([Buffer(input_buffer, cuda=enable_cuda) for _ in range(10)])
    decim = Decimate(input_buffer, demod_buffer, zero_phase=True, cuda=enable_cuda)
    demod = [demodulator(demod_buffer, audio_buffer, cuda=enable_cuda) for _ in tuner.channels]
    afile = [open("FM_{}.if32".format(int(f.center_frequency)), "bw") for f in tuner.channels]

    print("# Tuner Settings:")
    print("     Bandwidth: {}".format(tuner.input_bandwidth))
    print("     Mean Frequency: {}".format(tuner.input_frequency))
    print("     Stations: {}".format(len(tuner.channels)))

    # SoapySDR configuration.
    print("Configuring device...")
    sdr = Device({"driver": device_name})
    sdr.setGainMode(SOAPY_SDR_RX, 0, True)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, tuner.input_bandwidth)
    sdr.setFrequency(SOAPY_SDR_RX, 0, tuner.input_frequency)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    # Start collecting data.
    print("Starting device and audio stream...")
    sdr.activateStream(rx)
    stream = sd.OutputStream(blocksize=int(audio_buffer), callback=process,
                             samplerate=int(audio_rate), channels=demod.channels)

    try:
        print("Starting playback...")
        while True:
            receive(queue)
    except KeyboardInterrupt:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        sys.exit('\nInterrupted by user. Closing...')
