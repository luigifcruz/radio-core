"""Defines a Buffer module."""

import threading
from typing import Union
from contextlib import contextmanager

from radiocore._internal import Injector


class Buffer(Injector):
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

    def __init__(self, size: Union[int, float], dtype: str = "complex64",
                 lock: bool = False, cuda: bool = False):
        """Initialize the Buffer class."""
        self._lock: bool = lock
        self._cuda: bool = cuda
        self._dtype: str = dtype
        self._size: int = int(size)

        if self._lock:
            self._mtx = threading.Lock()

        super().__init__(self._cuda)

        if self._cuda:
            self._buffer = self._xs.get_shared_mem(self._size,
                                                   dtype=self._dtype)
        else:
            self._buffer = self._np.zeros(self._size, dtype=self._dtype)

    @property
    def dtype(self):
        """Return the dtype of the array."""
        return self._buffer.dtype

    @property
    def is_cuda(self) -> bool:
        """Return if the array is allocated in the GPU memory."""
        return self._cuda

    @property
    def size(self) -> int:
        """Return the size of the array."""
        return self._size

    @property
    def is_locked(self) -> bool:
        """Return if the array is currently in use."""
        if not self._lock:
            raise ValueError("locking is not enabled in this instance")
        return self._mtx.locked()

    @contextmanager
    def consume(self):
        """
        Return a handle of the original array memory.

        When lock is enabled, it will also restrict the access to this
        resource until the original caller is done.
        """
        try:
            if self._lock:
                self._mtx.acquire()
            yield self._buffer
        finally:
            if self._lock:
                self._mtx.release()
