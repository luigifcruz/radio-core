"""Defines a WBFM module."""

from typing import Union
from radiocore._internal import Injector
from radiocore.analog import PLL
from radiocore.analog.fm import FM
from radiocore.analog.deemphasis import Deemphasis
from radiocore.analog.decimate import Decimate
from radiocore.analog.bandpass import Bandpass


class WBFM(Injector):
    """
    The WBFM class provides a Stereo demodulator for FM signals.

    For mono FM-stations, use the MFM class.
    For simple FM demodulation, use the FM class.

    Parameters
    ----------
    input_size : int, float
        input signal buffer size
    output_size : int, float
        output signal buffer size
    deemphasis_rate: float
        audio deemphasis rate, 75e-6 for americas,
        otherwise 50e-6 (default is 75e-6)
    cuda : bool
        use the GPU for processing (default is False)
    """

    def __init__(self,
                 input_size: Union[int, float],
                 output_size: Union[int, float],
                 deemphasis_rate: float = 75e-6,
                 cuda: bool = False):
        """Initialize the Stereo-FM class."""
        self._cuda: bool = cuda
        self._input_size: int = int(input_size)
        self._output_size: int = int(output_size)

        self._fm_demod = FM(self._input_size, self._input_size,
                            cuda=self._cuda)

        self._plt_filter = Bandpass(self._input_size, 19e3-100, 19e3+100,
                                    cuda=self._cuda)

        self._lmr_filter = Bandpass(self._input_size, 23e3, 53e3,
                                    cuda=self._cuda)

        self._pll = PLL(cuda=self._cuda)

        self._decimate = Decimate(self._input_size, self._output_size,
                                  zero_phase=True, cuda=self._cuda)

        self._left_deemphasis = Deemphasis(self._output_size, deemphasis_rate,
                                           cuda=self._cuda)

        self._right_deemphasis = Deemphasis(self._output_size, deemphasis_rate,
                                            cuda=self._cuda)

        super().__init__(cuda)

    @property
    def channels(self):
        """Return the number of audio channels of the output."""
        return 2

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
        _tmp = self._fm_demod.run(input_sig, False)

        # Filter pilot and update PLL.
        self._pll.step(self._plt_filter.run(_tmp))

        # Filter the Left - Right component.
        _lmr = self._lmr_filter.run(_tmp)
        _lmr = (self._pll.wave(2) * _lmr) * 1.0175

        # Mix L+R and L-R to generate L and R
        _l = self._decimate.run(_tmp + _lmr)
        _r = self._decimate.run(_tmp - _lmr)

        # Deemphasize channels.
        _l = self._left_deemphasis.run(_l)
        _r = self._right_deemphasis.run(_r)

        # Remove DC from signal.
        _l -= self._xp.mean(_l)
        _r -= self._xp.mean(_r)

        # Ensure Bounds
        _l = self._xp.clip(_l, -0.999, 0.999)
        _r = self._xp.clip(_r, -0.999, 0.999)

        if self._cuda and numpy_output:
            _l = self._xp.asnumpy(_l)
            _r = self._xp.asnumpy(_r)

        return _l, _r
