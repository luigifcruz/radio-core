import sys

import zmq
import numpy as np
import sounddevice as sd

from SoapySDR import Device, SOAPY_SDR_CF32, SOAPY_SDR_RX
from radiocore import Buffer, RingBuffer, FM, MFM, WBFM, Tuner

def process():
    radio_buffers = tuner.load(output_buffer.data)
    print("a")

    for i, c in enumerate(tuner.channels):
        demoded = demod[i].run(tuner.run(i))
        print(demoded.dtype)

        address = int(c.center_frequency).to_bytes(4, byteorder='little')
        socket.send_multipart([address, demoded.tobytes()])

def receive():
    sdr.readStream(rx, [input_buffer.data], device_rate, timeoutUs=2**37)
    ring_buffer.append(input_buffer.data)

    if ring_buffer.popleft(output_buffer.data):
        process()

enable_cuda: bool = False        # If True, enable CUDA demodulation.
deemphasis: float = 75e-6       # 50e-6 for World and 75e-6 for Americas and S. Korea.
input_rate: float = 10e6        # SDR RX bandwidth.
demod_rate: float = 250e3       # FM station bandwidth. (240-256 kHz).
audio_rate: float = 48e3        # Audio bandwidth (32-48 kHz).
device_rate: int = 1024         # Device buffer size.
device_name: str = "airspy"     # SoapySDR device string.
demodulator = FM              # Demodulator (WBFM, MFM, or FM).

# Setup ZeroMQ server.
context = zmq.Context()
context.setsockopt(zmq.IPV6, True)
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

# Create Tuner and channels to be demoded.
tuner = Tuner(cuda=enable_cuda)
tuner.add_channel(94.5e6, demod_rate)
tuner.add_channel(97.5e6, demod_rate)
tuner.add_channel(96.9e6, demod_rate)
tuner.request_bandwidth(input_rate)

# Start DSP processors.
print("Configuring DSP...")
demod = [demodulator(demod_rate, audio_rate, cuda=enable_cuda) for _ in tuner.channels]

# Allocate buffers.
print("Allocating buffers...")
ring_buffer = RingBuffer(input_rate * 10, dtype=np.complex64, cuda=enable_cuda)
input_buffer = Buffer(device_rate, dtype=np.complex64, cuda=enable_cuda)
output_buffer = Buffer(input_rate, dtype=np.complex64, cuda=enable_cuda)

# SoapySDR configuration.
print("Configuring device...")
sdr = Device({"driver": device_name})
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, tuner.input_bandwidth)
sdr.setFrequency(SOAPY_SDR_RX, 0, tuner.input_frequency)
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

# Start collecting data.
print("Starting device...")
sdr.activateStream(rx)

try:
    print("Starting playback...")
    while True: receive()
except KeyboardInterrupt:
    sdr.deactivateStream(rx)
    sdr.closeStream(rx)
    sys.exit('\nInterrupted by user. Closing...')
