"""Defines a PLL module."""

from radiocore._internal import Injector


class PLL(Injector):
    """
    The PLL class creates a phase-locked signal from an input signal.

    This is usefull to change the frequency of a pilot signal.
    This class is based in the Hilbert transform.

    Attributes
    ----------
    cuda : bool, optional
        use the GPU for processing (default is False)
    """

    def __init__(self, cuda=False):
        """Initialize the PLL class."""
        self._cuda = cuda
        self._baseline = None
        super().__init__(self._cuda)

    def step(self, input_sig):
        """Update the internal state according to the input_sig (arr)."""
        self._baseline = self._xs.hilbert(input_sig)

    def wave(self, mult=1.0):
        """Return the phase-locked signal multiplied by mult."""
        return self._xp.real(self._baseline**mult)
