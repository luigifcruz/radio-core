import cupy as cp
from typing import List


def frequency(x, fs) -> float:
    zeros = cp.where(cp.diff(cp.signbit(x)))[0]
    return fs/cp.mean(cp.diff(zeros[5:-5]))/2


def alignment(x, algn, freq, times) -> List[float]:
    err_ls = []
    for i in range(len(algn)):
        signal = x + cp.sin(2.0*cp.pi*freq*times+algn[i])
        err_ls.append(cp.mean(cp.abs(signal)))
    return cp.asarray(err_ls)
