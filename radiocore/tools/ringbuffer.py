"""Defines a Ring Buffer module."""

import atomics
from threading import Event
from typing import Union

from radiocore._internal import Injector


class RingBuffer(Injector):
    """
    The Ring Buffer class manage a CPU or GPU array in a circular fashion.

    The CUDA (GPU) backbuffer is allocated by cuSignal. The memory is
    managed. Therefore, it's DMA'ed to the CPU automatically.

    Parameters
    ----------
    capacity : int, float
        maximum capacity of the backbuffer
    dtype : str, optional
        element type of the array (default is complex64)
    cuda : bool, optional
        allocate memory on the GPU (default is False)
    print_overflow : bool, optional
        print to stdout if buffer overflow happens (default is True)
    allow_overflow : bool, optional
        let overflow happen without raising an exception (default is True)
    """

    def __init__(self,
                 capacity: Union[int, float],
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
        self._cv = Event()
        self._head: int = 0
        self._tail: int = 0
        self._occupancy = atomics.atomic(width=4, atype=atomics.INT)

        super().__init__(self._cuda)

        if self._cuda:
            self._buffer = self._xs.get_shared_mem(self._capacity,
                                                   dtype=self._dtype)
        else:
            self._buffer = self._np.zeros(self._capacity, dtype=self._dtype)

    @property
    def capacity(self) -> int:
        """Return buffer capacity."""
        return self._capacity

    @property
    def occupancy(self) -> int:
        """Return the current buffer occupancy. Used space."""
        return self._occupancy.load()

    @property
    def vacancy(self) -> int:
        """Return the current buffer vacancy. Space left."""
        return self.capacity - self.occupancy

    @property
    def data(self):
        """Return the backbuffer. Use with care."""
        return self._buffer

    def reset(self):
        """Reset ringbuffer state."""
        self._tail = 0
        self._head = 0
        self._occupancy.store(0)

    def __str__(self) -> str:
        """Return printable version of the backbuffer."""
        return self._buffer.__str__()

    def __copy(self, dst, src, size):
        if size == 0:
            return

        if size < 0:
            raise ValueError(f"Copy size is negative! ({size})")

        dst[:size] = src[:size]

    def put(self, buffer):
        """
        Copy all buffer elements into ring buffer.

        Parameters
        ----------
        buffer : ndarray
            array containing the elements to be copied
        """
        _size: int = len(buffer)

        if _size > self.capacity:
            raise ValueError("Input buffer is bigger than ring capacity.")

        if _size > self.vacancy:
            if not self._allow_overflow:
                raise ValueError("Overflow happened.")

            if self._print_overflow:
                print("overflow")

            self.reset()

        _copy_len_a = min(_size, self.capacity - self._head)
        _copy_len_b = _size - _copy_len_a if (_copy_len_a < _size) else 0

        self.__copy(self._buffer[self._head:], buffer, _copy_len_a)
        self.__copy(self._buffer, buffer[_copy_len_a:], _copy_len_b)

        self._head = (self._head + _size) % self.capacity
        self._occupancy.add(_size)

        self._cv.set()

    def get(self, buffer, timeout: float = 3.0):
        """
        Fill all buffer elements with the ring buffer data.

        Parameters
        ----------
        buffer : ndarray
            array where the elements will be copied into
        timeout : float, optional
            how long in seconds the function should wait (default is 3)
        """
        _size: int = len(buffer)

        if _size > self.capacity:
            raise ValueError("Input buffer is bigger than ring capacity.")

        while _size > self.occupancy:
            if not self._cv.wait(timeout):
                # Timeout reached. Retuning.
                return
            self._cv.clear()

        _copy_len_a = min(_size, self.capacity - self._tail)
        _copy_len_b = _size - _copy_len_a if (_copy_len_a < _size) else 0

        self.__copy(buffer, self._buffer[self._tail:], _copy_len_a)
        self.__copy(buffer[_copy_len_a:], self._buffer, _copy_len_b)

        self._tail = (self._tail + _size) % self.capacity
        self._occupancy.sub(_size)

        return True
