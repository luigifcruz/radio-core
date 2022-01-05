"""Defines a signal decimator."""

from typing import Union
from radiocore._internal import Injector


class Decimate(Injector):
    """
    The Decimate class provides FIR decimation.

    Parameters
    ----------
    input_size : int, float
        input signal buffer size
    output_size : int, float
        output signal buffer size
    zero_phase : bool
        decimate with zero phase using filtfilt (default is True)
    cuda : bool
        use the GPU for processing (default is False)
    """

    def __init__(self,
                 input_size: Union[int, float],
                 output_size: Union[int, float],
                 zero_phase: bool = True,
                 cuda: bool = False):
        """Initialize the Decimate class."""
        self._cuda: bool = cuda
        self._phase: bool = zero_phase
        self._input_size: int = int(input_size)
        self._output_size: int = int(output_size)

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
        _tmp = self._xs.resample(_tmp, self._output_size, window="hann")

        return _tmp
