"""Imports all modules from radiocore."""

from radiocore.analog import *
from radiocore.tools import *

def HasCuda():
    r"""
    Check if the system has the modules needed for the GPU acceleration.

    Note
    ----
    Modules are listed on `requirements_gpu.txt`.

    Returns
    -------
    has_gpu : bool
        True if the system has GPU capabilities.
    """
    try:
        import cupy
        cupy.__version__
        import cusignal
        cusignal.__version__
    except:
        return False
    return True

__version__ = '1.0.0'
