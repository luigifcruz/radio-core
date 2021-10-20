"""Defines a Chopper module."""


class Chopper:
    """
    The Copper class is a helper to divide a big array into smaller chunks.

    It's usefull when you need to populate an array used for processing with
    smaller arrays.

    Attributes
    ----------
    size : int, float
        total size of the original array
    chunk_size : int, float
        desired chunk size
    """

    def __init__(self, size, chunk_size):
        """Initialize the Chopper class."""
        self._size = int(size)
        self._chunk_size = int(chunk_size)

        if (self._size % self._chunk_size) != 0:
            raise ValueError("cannot evenly divide array by chunk size "
                             f"({self._size}, {self._chunk_size})")

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
