import cupy as cp


def frequency(x, fs):
    zeros = cp.where(cp.diff(cp.signbit(x)))[0]
    return fs/cp.mean(cp.diff(zeros[5:-5]))/2


def alignment(x, algn, freq, times):
    err_ls = []
    for i in range(len(algn)):
        signal = x + cp.sin(2.0*cp.pi*freq*times+algn[i])
        err_ls.append(cp.mean(cp.abs(signal)))
    return cp.asarray(err_ls)
