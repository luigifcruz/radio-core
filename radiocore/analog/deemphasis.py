"""Defines a deemphasis filter for FM signals."""

from typing import Union
from radiocore._internal import Injector


class Deemphasis(Injector):
    """
    The Deemphasis class provides FM broadcast deemphasis.

    This class is internally used by the WBFM and MFM classes.

    Parameters
    ----------
    input_size : int, float
        input signal buffer size
    rate: float
        audio deemphasis rate, 75e-6 for americas,
        otherwise 50e-6 (default is 75e-6)
    dtype: str
        type of the output signal (default is float32)
    cuda : bool
        use the GPU for processing (default is False)
    """

    def __init__(self, input_size: Union[int, float], rate: float = 75e-6,
                 dtype: str = "float32", cuda: bool = False):
        """Initialize the Deemphasis class."""
        self._cuda: bool = cuda
        self._dtype: str = dtype
        self._rate: float = rate
        self._input_size: int = int(input_size)

        super().__init__(cuda)

        # Generate IIR taps for the deemphasis filter.
        _x = self._np.exp(-1/(self._input_size * self._rate))
        _b = ([1 - _x], [1, -_x])

        # Convert IIR taps to FIR. This improves processing time on the GPU.
        _c = self._ss.dlti(*_b)
        _, _d = self._ss.dimpulse(_c, n=51)
        _b = self._np.squeeze(_d)
        _b = self._xp.array(_b, dtype=self._dtype)
        _a = self._xp.array(1.0, dtype=self._dtype)
        self._taps = (_b, _a)

        _zi = self._xs.lfilter_zi(*self._taps)
        self._state = self._xp.array(_zi, dtype=self._dtype)

    def run(self, input_sig):
        """
        Deemphasizes the input signal and output the buffer.

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
