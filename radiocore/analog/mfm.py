"""Defines a broadcast Mono-FM demodulator module."""

from typing import Union
from radiocore._internal import Injector
from radiocore.analog.fm import FM
from radiocore.analog.deemphasis import Deemphasis


class MFM(Injector):
    """
    The MFM class provides a Mono demodulator for FM signals.

    For stereo FM-stations, use the WBFM class.
    For simple FM demodulation, use the FM class.

    Parameters
    ----------
    input_size : int, float
        input signal buffer size
    output_size : int, float
        output signal buffer size
    deemphasis: float
        audio deemphasis rate, 75e-6 for americas,
        otherwise 50e-6 (default is 75e-6)
    cuda : bool
        use the GPU for processing (default is False)
    """

    def __init__(self,
                 input_size: Union[int, float],
                 output_size: Union[int, float],
                 deemphasis: float = 75e-6,
                 cuda: bool = False):
        """Initialize the Mono-FM class."""
        self._cuda: bool = cuda
        self._input_size: int = int(input_size)
        self._output_size: int = int(output_size)

        self._fm_demod = FM(self._input_size, self._output_size,
                            cuda=self._cuda)
        self._deemphasis = Deemphasis(self._output_size, deemphasis,
                                      cuda=self._cuda)

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
        _tmp = self._fm_demod.run(input_sig, False)[:, 0]
        _tmp = self._deemphasis.run(_tmp)
        _tmp -= self._xp.mean(_tmp)
        _tmp = self._xp.clip(_tmp, -0.999, 0.999)
        _tmp = self._xp.expand_dims(_tmp, axis=1)

        if self._cuda and numpy_output:
            return self._xp.asnumpy(_tmp)

        return _tmp
