"""Defines a Chopper module."""


class Chopper:
    """
    The Copper class is a helper tool to divide a big array into smaller chunks.

    It's usefull when you need to populate an array used for processing with
    smaller arrays.

    Attributes
    ----------
    size : int
        total size of the original array
    chunk_size : int
        desired chunk size
    """

    def __init__(self, size: int, chunk_size: int):
        """Initialize the Chopper class."""
        self._size = size
        self._chunk_size = chunk_size

        if (self._size % self._chunk_size) != 0:
            raise ValueError("cannot evenly divide array by chunk size")

    def chop(self, input_arr):
        """
        Return an array of references to the original memory each with the set size.

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
