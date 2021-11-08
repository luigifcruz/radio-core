"""Defines a Chopper module."""

from typing import Union


class Chopper:
    """
    The Copper class is a helper to divide a big array into smaller chunks.

    It's usefull when you need to populate an array used for processing with
    smaller arrays.

    Parameters
    ----------
    size : int, float
        total size of the original array
    chunk_size : int, float
        desired chunk size
    """

    def __init__(self, size: Union[int, float], chunk_size: Union[int, float]):
        """Initialize the Chopper class."""
        self._size: int = int(size)
        self._chunk_size: int = int(chunk_size)

        if (self._size % self._chunk_size) != 0:
            raise ValueError("cannot evenly divide array by chunk size "
                             f"({self._size}, {self._chunk_size})")

    @property
    def size(self):
        """Return the size of the entire buffer."""
        return self._size

    @property
    def chunk_size(self):
        """Return the chunk size."""
        return self._chunk_size

    def chop(self, input_arr):
        """
        Return a reference to the bigger array's original memory.

        Parameters
        ----------
        input_arr : arr
            original array
        """
        for i in range(self._size//self._chunk_size):
            yield input_arr[self._chunk_size*i:self._chunk_size*(i+1)]

    @staticmethod
    def get_to_da_choppa():
        """Return where is the choppa."""
        return 'https://www.youtube.com/watch?v=Xs_OacEq2Sk'
