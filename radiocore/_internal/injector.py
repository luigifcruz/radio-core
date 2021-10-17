"""Defines a Injector module."""

import importlib


class Injector:
    """
    The Injector internal class ynamically loads and injects the supported
    modules into self.

    Attributes
    ----------
    cuda : bool
        enables GPU modules
    """

    def __init__(self, cuda=False):
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