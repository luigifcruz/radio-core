import sys

import numpy as np
import sounddevice as sd

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, RingBuffer, FM, MFM, WBFM, Tuner

enable_cuda: bool = False       # If True, enable CUDA demodulation.
deemphasis: float = 75e-6       # 50e-6 for World and 75e-6 for Americas and S. Korea.
input_rate: float = 10e6        # SDR RX bandwidth.
demod_rate: float = 250e3       # FM station bandwidth. (240-256 kHz).
audio_rate: float = 48e3        # Audio bandwidth (32-48 kHz).
device_rate: int = 1024         # Device buffer size.
device_name: str = "airspy"     # SoapySDR device string.
demodulator = WBFM              # Demodulator (WBFM, MFM, or FM).

# Create Tuner and channels to be demoded.
tuner = Tuner(cuda=enable_cuda)
tuner.add_channel(94.5e6, demod_rate)
tuner.add_channel(97.5e6, demod_rate)
tuner.add_channel(96.9e6, demod_rate)
tuner.request_bandwidth(input_rate)

# Queue and shared memory allocation.
print("Configuring DSP...")
demod = [demodulator(demod_rate, audio_rate, cuda=enable_cuda) for _ in tuner.channels]
ring_buffer = RingBuffer(input_rate * 10, dtype=np.complex64, cuda=enable_cuda)
output_buffer = Buffer(input_rate, dtype=np.complex64, cuda=enable_cuda)

# SoapySDR configuration.
print("Configuring device...")
sdr = Device({"driver": device_name})
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, tuner.input_bandwidth)
sdr.setFrequency(SOAPY_SDR_RX, 0, tuner.input_frequency)
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)


# Define demodulation callback. This should not block.
def process(outdata, *_):
    if not ring_buffer.popleft(output_buffer.data):
        return

    radio_buffers = tuner.load(output_buffer.data)

    for i, _ in enumerate(tuner.channels):
        demoded = demod[i].run(tuner.run(i))

    if demod[i].channels == 2:
        outdata[:] = np.dstack(demoded)
    else:
        outdata[:, 0] = demoded


# Start collecting data.
print("Starting device and audio stream...")
sdr.activateStream(rx)
stream = sd.OutputStream(blocksize=int(audio_rate), callback=process,
                         samplerate=int(audio_rate), channels=demod[0].channels)

try:
    print("Starting playback...")
    input_buffer = Buffer(device_rate, dtype=np.complex64, cuda=enable_cuda)

    while True:
        # Fill input buffer with complex data.
        sdr.readStream(rx, [input_buffer.data], device_rate, timeoutUs=2**37)

        # Append data to ring buffer.
        ring_buffer.append(input_buffer.data)

        # Start audio when buffer reaches N samples.
        if ring_buffer.occupancy > input_rate and not stream.active:
            stream.start()

except KeyboardInterrupt:
    sdr.deactivateStream(rx)
    sdr.closeStream(rx)
    sys.exit('\nInterrupted by user. Closing...')
