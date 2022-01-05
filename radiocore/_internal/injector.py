"""Defines a Injector module."""

import importlib
import pyfftw
import scipy

class Injector:
    """
    The Injector class dynamically loads and injects the modules into self.

    Attributes
    ----------
    cuda : bool
        enables GPU modules
    """

    def __init__(self, cuda=False):
        """Initialize the Injector class."""
        if cuda:
            self._xs = importlib.import_module('cusignal')
            self._xp = importlib.import_module('cupy')
            self._np = importlib.import_module('numpy')
            self._ss = importlib.import_module('scipy.signal')
        else:
            self._xs = importlib.import_module('scipy.signal')
            self._xp = importlib.import_module('numpy')
            self._np = self._xp
            self._ss = self._xs

        pyfftw.config.NUM_THREADS = 4
        scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)
        pyfftw.interfaces.cache.enable()
