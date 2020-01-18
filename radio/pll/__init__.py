import collections
import importlib
import numpy as np

class PLL:
  def __init__(self, fs, length, cuda=False):
    self.cuda = cuda

    if self.cuda:
      self.xp = importlib.import_module('cupy')
      self.np = importlib.import_module('numpy')
      self.ts = importlib.import_module('radio.pll._tools_cuda')
    else:
      self.xp = importlib.import_module('numpy')
      self.ts = importlib.import_module('radio.pll._tools')
      self.np = self.xp 

    self.fs = fs
    self.freq = 0.0
    self.len = length
    self.f_cb = collections.deque(maxlen=128)
    self.algn = self.xp.arange(0, self.xp.pi*2, self.xp.pi/6)
    self.times = self.xp.arange(0, 1, 1/fs)[:self.len]

    print("[PLL] Compiling Numba...")
    signal = self.wave(self.fs/4, 1.0)
    result = self.ts.frequency(signal, self.fs)
    phase = self.ts.alignment(signal, self.algn, self.fs/4, self.times)
    print("[PLL] Done ({}, {}, {})".format(self.fs/4, result, self.xp.argmax(phase)))

  def wave(self, freq, phase):
    return self.xp.cos(2.0*self.xp.pi*freq*self.times+phase)
      
  def step(self, x):
    # Estimate Signal Frequency
    freq = self.ts.frequency(x, self.fs)
    self.f_cb.append(freq)
    self.freq = self.np.mean(self.f_cb)

    # Get Phase Compensation
    error_ls = self.ts.alignment(x, self.algn, self.freq, self.times)
    self.phi = self.algn[self.xp.argmax(error_ls)]
      
  def mult(self, mult=1):
    omega = ((self.phi*mult)+(self.xp.pi/mult))
    return self.wave(self.freq*mult, omega)