import collections
import importlib

class PLL:
  def __init__(self, fs, length, cuda=False):
    self.cuda = cuda

    if self.cuda:
      self.xp = importlib.import_module('cupy')
      self.np = importlib.import_module('numpy')
    else:
      self.xp = importlib.import_module('numpy')
      self.np = self.xp 

    self.fs = fs
    self.freq = 0.0
    self.len = length
    self.f_cb = collections.deque(maxlen=128)
    self.algn = self.xp.arange(0, self.xp.pi*2, self.xp.pi/6)
    self.times = self.xp.arange(0, 1, 1/fs)[:self.len]

    print("[PLL] Compiling Numba...")
    signal = self.wave(self.xp, self.fs/4, 1.0, self.times)
    result = self.freq_estimator(self.xp, signal, self.fs)
    phase = self.alignments(self.xp, signal, self.algn, self.fs/4, self.times)
    print("[PLL] Done ({}, {}, {})".format(self.fs/4, result, self.xp.argmax(phase)))

  @staticmethod
  def wave(xp, freq, phase, times):
    return xp.cos(2.0*xp.pi*freq*times+phase)

  @staticmethod
  def freq_estimator(xp, x, fs):
    zeros = xp.where(xp.diff(xp.signbit(x)))[0]
    return fs/xp.mean(xp.diff(zeros[5:-5]))/2

  @staticmethod
  def alignments(xp, x, algn, freq, times):
    err_ls = []
    for i in range(len(algn)):
        signal = x + xp.sin(2.0*xp.pi*freq*times+algn[i])
        err_ls.append(xp.mean(xp.abs(signal)))
    return xp.asarray(err_ls)
      
  def step(self, x):
    # Estimate Signal Frequency
    freq = self.freq_estimator(self.xp, x, self.fs)
    self.f_cb.append(freq)
    self.freq = self.np.mean(self.f_cb)

    # Get Phase Compensation
    error_ls = self.alignments(self.xp, x, self.algn, self.freq, self.times)
    self.phi = self.algn[self.xp.argmax(error_ls)]
      
  def mult(self, mult=1):
    omega = ((self.phi*mult)+(self.xp.pi/mult))
    return self.wave(self.xp, self.freq*mult, omega, self.times)