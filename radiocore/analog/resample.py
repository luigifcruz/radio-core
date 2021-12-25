"""Defines a signal resampler (decimation + filtering)."""

from typing import Union
from radiocore._internal import Injector


class Resample(Injector):
    """
    The Resample class provides FIR decimation and filtering.

    Parameters
    ----------
    input_size : int, float
        input signal buffer size
    output_size : int, float
        output signal buffer size
    cuda : bool
        use the GPU for processing (default is False)
    """

    def __init__(self,
                 input_size: Union[int, float],
                 output_size: Union[int, float],
                 cuda: bool = False):
        """Initialize the Resample class."""
        self._cuda: bool = cuda
        self._input_size: int = int(input_size)
        self._output_size: int = int(output_size)

        self._rate: int = self._input_size // self._output_size

        super().__init__(cuda)

    def run(self, input_sig):
        """
        Decimate the input signal and output the result.

        Parameters
        ----------
        input_sig : arr
            input signal array, size should match the input_size
        """
        if len(input_sig) != self._input_size:
            raise ValueError("input_sig size and input_size mismatch")

        _tmp = self._xp.asarray(input_sig)

        if self._rate != 1:
            _tmp = self._xs.resample_poly(_tmp, 1, self._rate)

        _tmp = self._xs.resample(_tmp, self._output_size)

        if len(_tmp) != self._output_size:
            raise EnvironmentError("output size and output_size mismatch")

        return _tmp
