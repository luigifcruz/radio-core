"""Defines a Ring Buffer module."""

from threading import Condition
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
        self._cv = Condition()

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
        """Return buffer capacity."""
        return self._capacity

    @property
    def occupancy(self) -> int:
        """Return the current buffer occupancy. Used space."""
        return self._occupancy

    @property
    def vacancy(self) -> int:
        """Return the current buffer vacancy. Space left."""
        return self.capacity - self.occupancy

    @property
    def data(self):
        """Return the backbuffer. Use with care."""
        return self._buffer

    def __str__(self) -> str:
        """Return printable version of the backbuffer."""
        return self._buffer.__str__()

    def __copy(self, dst, src):
        return self._xp.copyto(self._xp.asarray(dst), self._xp.asarray(src))

    def append(self, buffer):
        """
        Copy all buffer elements into ring buffer.

        Parameters
        ----------
        buffer : ndarray
            array containing the elements to be copied
        """
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
            self._buffer[self._head:self._head+len(buffer)] = buffer
        else:
            _remainer = self.capacity - self._head
            self._buffer[self._head:self.capacity] = buffer[:_remainer]
            self._buffer[:len(buffer)-_remainer] = buffer[_remainer:len(buffer)]

        self._head = (self._head + len(buffer)) % self.capacity
        self._occupancy = self.occupancy + len(buffer)

        with self._cv:
            self._cv.notify_all()

    def popleft(self, buffer):
        """
        Fill all buffer elements with the ring buffer data.

        Parameters
        ----------
        buffer : ndarray
            array where the elements will be copied into
        """
        if len(buffer) > self.capacity:
            raise ValueError("Input buffer is bigger than ring capacity.")

        with self._cv:
            while len(buffer) > self.occupancy:
                self._cv.wait()

        if (self.capacity - self._tail) >= len(buffer):
            _src = self._buffer[self._tail:self._tail+len(buffer)]
            _dst = buffer
            self.__copy(_dst, _src)
        else:
            _remainer = self.capacity - self._tail

            _src = self._buffer[self._tail:self.capacity]
            _dst = buffer[:_remainer]
            self.__copy(_dst, _src)

            _src = self._buffer[:len(buffer)-_remainer]
            _dst = buffer[_remainer:]
            self.__copy(_dst, _src)

        self._tail = (self._tail + len(buffer)) % self.capacity
        self._occupancy = self.occupancy - len(buffer)

        return True
