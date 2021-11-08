"""Defines a bandpass filter."""

from typing import Union
from radiocore._internal import Injector


class Bandpass(Injector):
    """
    The Bandpass class provides a zero-phase bandpass filter..

    Parameters
    ----------
    input_size : int, float
        input signal buffer size
    start_freq : int, float
        start of the bandpass window, value in Hz
    stop_freq : int, float
        end of the bandpass window, value in Hz
    num_taps : int
        number of filter taps (default is 51)
    window : str
        window filter function (default is hann)
    cuda : bool
        use the GPU for processing (default is False)
    """

    def __init__(self,
                 input_size: Union[int, float],
                 start_freq: Union[int, float],
                 stop_freq: Union[int, float],
                 num_taps: int = 51,
                 window: str = "hann",
                 cuda: bool = False):
        """Initialize the Bandpass class."""
        self._cuda: bool = cuda
        self._window: str = window
        self._num_taps: int = int(num_taps)
        self._input_size: int = int(input_size)
        self._stop_freq: float = float(stop_freq)
        self._start_freq: float = float(start_freq)

        super().__init__(cuda)

        _lo = self.__nyq(self._start_freq)
        _hi = self.__nyq(self._stop_freq)
        _tp = self._xs.firwin(self._num_taps, [_lo, _hi], pass_zero=False,
                              window=self._window)

        self._taps = (_tp, 1.0)
        self._state = self._xs.lfilter_zi(*self._taps)

    def __nyq(self, freq_hz):
        return (freq_hz / (0.5 * self._input_size))

    def run(self, input_sig):
        """
        Filter the input signal and output the result.

        Parameters
        ----------
        input_sig : arr
            input signal array, size should match the input_size
        """
        if len(input_sig) != self._input_size:
            raise ValueError("input_sig size and input_size mismatch")

        _tmp = self._xp.asarray(input_sig)
        _tmp, self._state = self._xs.lfilter(*self._taps, _tmp, zi=self._state)

        return _tmp
