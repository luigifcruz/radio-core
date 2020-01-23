import numpy as np
from typing import List


def frequency(x, fs) -> float:
    zeros = np.where(np.diff(np.signbit(x)))[0]
    return float(fs/np.mean(np.diff(zeros[5:-5]))/2)


def alignment(x, algn, freq, times) -> List[float]:
    err_ls = []
    for i in range(len(algn)):
        signal = x + np.sin(2.0*np.pi*freq*times+algn[i])
        err_ls.append(np.mean(np.abs(signal)))
    return np.asarray(err_ls)
