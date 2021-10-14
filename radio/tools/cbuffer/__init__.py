from contextlib import contextmanager
import collections
import importlib

# This CircularBuffer is cursed. It doesn't lock. It isn't thread safe.
# I'm obligated to use it because collections.queue is crap for real-time stuff.
# This is bad but it's very cheap. Reduced mallocs and memcpys.

class CBuffer:
    occupancy: int = 0
    tail: int = 0
    head: int = 0
    capacity: int


    def __init__(self, capacity, writeSize=2000, dtype='complex64', cuda=False):
        self.dtype = dtype
        self.capacity = capacity
        self.writeSize = writeSize

        self.load_modules(cuda)

        if cuda:
            self.buffer = self.sg.get_shared_mem(self.capacity + self.writeSize, dtype=self.dtype)
        else:
            self.buffer = self.np.zeros(self.capacity + self.writeSize, dtype=self.dtype)


    def load_modules(self, cuda):
        self.cuda = cuda

        if self.cuda:
            self.sg = importlib.import_module('cusignal')
        else:
            self.np = importlib.import_module('numpy')


    @contextmanager
    def write(self, size):
        if size > self.capacity:
            raise ValueError('write size beyond capacity')

        try:
            yield self.buffer[self.head:self.head+size]
        finally:
            self.head += size
            self.occupancy += size

            if self.head >= self.capacity:
                self.head = 0


    @contextmanager
    def read(self, size):
        if size > self.capacity:
            raise ValueError('read size beyond capacity')

        if self.occupancy < size:
            yield None; return

        try:
            yield self.buffer[self.tail:self.tail+size]
        finally:
            self.tail += size
            self.occupancy -= size

            if self.tail >= self.capacity:
                self.tail = 0


