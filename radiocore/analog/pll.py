"""Defines a PLL module."""

from radiocore._internal import Injector


class PLL(Injector):
    """
    The PLL class creates a phase-locked signal from an input signal.

    This is usefull to change the frequency of a pilot signal.
    This class is based in the Hilbert transform.

    Parameters
    ----------
    cuda : bool, optional
        use the GPU for processing (default is False)
    """

    def __init__(self, cuda: bool = False):
        """Initialize the PLL class."""
        self._cuda: bool = cuda
        self._baseline = None
        super().__init__(self._cuda)

    def step(self, input_sig):
        """
        Update the internal state according to the input_sig (arr).

        Parameters
        ----------
        input_sig : arr
            input signal array
        """
        self._baseline = self._xs.hilbert(input_sig)

    def wave(self, mult: float = 1.0):
        """
        Return the phase-locked signal multiplied by mult.

        Parameters
        ----------
        mult : int, float
            frequency multiplier of the output signal
        """
        _tmp = self._baseline ** mult
        return self._xp.real(_tmp) / self._xp.abs(_tmp)
