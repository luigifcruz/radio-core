import sys
import zmq
import queue
import numpy as np
import sounddevice as sd

frequency: float = 97.5e6             # Set the FM station frequency.
audio_rate: float = 48e3              # Audio bandwidth (32-48 kHz).
channels: float = 1                   # Number of audio channels (2 for Stereo).
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

# Define demodulation callback. This should not block.
def process(outdata, *_):
    if not que.empty():
        outdata[:] = que.get_nowait()[0]

# Configure sound device stream.
print("Starting audio stream...")
stream = sd.OutputStream(blocksize=int(audio_rate), callback=process,
                         samplerate=int(audio_rate), channels=channels)

try:
    print("Starting playback...")
    stream.start()

    while True:
        [_, payload] = socket.recv_multipart()
        audio = np.frombuffer(payload, dtype=np.float32)
        audio = audio.reshape((len(audio)//channels, channels))
        que.put_nowait([audio])

except KeyboardInterrupt:
    sys.exit('\nInterrupted by user. Closing...')
