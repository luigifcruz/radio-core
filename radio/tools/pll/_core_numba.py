import numpy as np
from numba import njit


@njit(fastmath=True)
def frequency(x, fs):
    zeros = np.where(np.diff(np.signbit(x)))[0]
    return fs/np.mean(np.diff(zeros[5:-5]))/2


@njit(fastmath=True)
def alignment(x, algn, freq, times):
    err_ls = []
    for i in range(len(algn)):
        signal = x + np.sin(2.0*np.pi*freq*times+algn[i])
        err_ls.append(np.mean(np.abs(signal)))
    return np.asarray(err_ls)
