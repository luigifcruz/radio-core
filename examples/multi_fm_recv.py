import sys

import zmq
import queue
import numpy as np
import sounddevice as sd


# Define demodulation callback. This should not block.
def process(outdata, *_):
    if que.empty():
        return

    outdata[:] = que.get_nowait()

def receive():
    [address, payload] = socket.recv_multipart()
    audio = np.frombuffer(payload, dtype=np.float32)
    que.put_nowait([audio])


frequency: float = 96.9e6             # Set the FM station frequency.
audio_rate: float = 48e3              # Audio bandwidth (32-48 kHz).
channels: float = 1                   #
server: str = "tcp://localhost:5555"  # Server address.

# Setup ZeroMQ client.
print("Creating ZeroMQ server...")
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(server)
address = int(frequency).to_bytes(4, byteorder='little')
socket.setsockopt(zmq.SUBSCRIBE, address)

# Allocate buffers.
print("Allocating buffers...")
que = queue.Queue()

# Start collecting data.
print("Starting device and audio stream...")
stream = sd.OutputStream(blocksize=int(audio_rate), callback=process,
                         samplerate=int(audio_rate), channels=channels)
stream.start()

try:
    print("Starting playback...")
    while True: receive()
except KeyboardInterrupt:
    sys.exit('\nInterrupted by user. Closing...')
