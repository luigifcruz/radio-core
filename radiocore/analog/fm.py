"""Defines a generic FM demodulator module."""

from typing import Union
from radiocore._internal import Injector
from radiocore.analog.decimate import Decimate


class FM(Injector):
    """
    The FM class provides a generic demodulator for FM signals.

    For broadcast FM stations, use the MFM for mono or WBFM for stereo.

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
        """Initialize the FM class."""
        self._cuda: bool = cuda
        self._input_size: int = int(input_size)
        self._output_size: int = int(output_size)

        self._decimate = Decimate(self._input_size, self._output_size,
                                  zero_phase=True, cuda=self._cuda)

        super().__init__(cuda)

    @property
    def channels(self):
        """Return the number of audio channels of the output."""
        return 1

    def run(self, input_sig, numpy_output: bool = True):
        """
        Demodulate the input signal and output the audio buffer.

        Parameters
        ----------
        input_sig : arr
            input signal array, size should match the input_size
        numpy_output: bool
            copy buffer to the cpu if cuda is enabled (default True)
        """
        if len(input_sig) != self._input_size:
            raise ValueError("input_sig size and input_size mismatch")

        _tmp = self._xp.asarray(input_sig)
        _tmp -= self._xp.mean(_tmp)
        _tmp = self._xp.angle(_tmp)
        _tmp = self._xp.unwrap(_tmp)
        _tmp = self._xp.diff(_tmp)
        _tmp = self._xp.concatenate((_tmp, self._xp.array([0])))
        _tmp = _tmp / self._xp.pi
        _tmp = self._decimate.run(_tmp)

        if self._cuda and numpy_output:
            return self._xp.asnumpy(_tmp)

        return _tmp
