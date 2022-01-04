"""Defines a Ring Buffer module."""

from typing import Union
from contextlib import contextmanager

from radiocore._internal import Injector

class RingBuffer(Injector):
    """
    The Buffer class manage a CPU or GPU array.

    The CUDA (GPU) array is allocated by cuSignal. The memory is
    manage. Therefore, it's DMA'ed to the CPU automatically.

    Parameters
    ----------
    size : int, float
        size of the array
    dtype : str, optional
        element type of the array (default is complex64)
    lock : bool, optional
        lock array when using it (default if False)
    cuda : bool, optional
        allocate memory on the GPU (default is False)
    """

    def __init__(self,
                 capacity: int,
                 dtype: str = "complex64",
                 cuda: bool = False,
                 print_overflow: bool = True,
                 allow_overflow: bool = True):
        """Initialize the Ring Buffer class."""
        self._print_overflow: bool = print_overflow
        self._allow_overflow: bool = allow_overflow
        self._capacity: int = int(capacity)
        self._cuda: bool = cuda
        self._dtype = dtype

        self._occupancy: int = 0
        self._head: int = 0
        self._tail: int = 0

        super().__init__(self._cuda)

        if self._cuda:
            self._buffer = self._xs.get_shared_mem(self._capacity,
                                                   dtype=self._dtype)
        else:
            self._buffer = self._np.zeros(self._capacity, dtype=self._dtype)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def occupancy(self) -> int:
        return self._occupancy

    @property
    def vacancy(self) -> int:
        return self.capacity - self.occupancy

    @property
    def data(self):
        return self._buffer

    def __str__(self) -> str:
        return self._buffer.__str__()

    def append(self, buffer):
        if len(buffer) > self.capacity:
            raise ValueError("Input buffer is bigger than ring capacity.")

        if len(buffer) > self.vacancy:
            if not self._allow_overflow:
                raise ValueError("Overflow happened.")

            if self._print_overflow:
                print("overflow")

            self._tail += len(buffer) - self.occupancy
            self._occupancy += len(buffer) - self.occupancy

        if (self.capacity - self._head) >= len(buffer):
            self._buffer[self._head:self._head+len(buffer)] = buffer;
        else:
            _remainer = self.capacity - self._head
            self._buffer[self._head:self.capacity] = buffer[:_remainer]
            self._buffer[:len(buffer)-_remainer] = buffer[_remainer:len(buffer)]

        self._head = (self._head + len(buffer)) % self.capacity;
        self._occupancy = self.occupancy + len(buffer)

    def popleft(self, buffer):
        if len(buffer) > self.capacity:
            raise ValueError("Input buffer is bigger than ring capacity.")

        if len(buffer) > self.occupancy:
            return False

        if (self.capacity - self._tail) >= len(buffer):
            _src = self._buffer[self._tail:self._tail+len(buffer)]
            _dst = buffer
            self._xp.copyto(_dst, _src)
        else:
            _src = self._buffer[self._tail:self.capacity]
            _dst = buffer
            self._xp.copyto(_dst, _src)

            _remainer = self.capacity - self._tail
            _src = self._buffer[:len(buffer)-_remainer]
            _dst = buffer[_remainer:]
            self._xp.copyto(_dst, _src)

        self._tail = (self._tail + len(buffer)) % self.capacity;
        self._occupancy = self.occupancy - len(buffer)

        return True
