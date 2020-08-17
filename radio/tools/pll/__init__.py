import collections
import importlib


class PLL:

    def __init__(self, fs, cuda=False):
        # Import Dynamic Modules
        self.load_modules(cuda)

        # Variables to Self
        self.fs = fs
        self.freq = 0.0
        self.times = None
        self.f_cb = collections.deque(maxlen=128)
        self.algn = self.xp.arange(0, self.xp.pi*2, self.xp.pi/4)

    def load_modules(self, cuda):
        self.cuda = cuda

        if self.cuda:
            self.xp = importlib.import_module('cupy')
            self.np = importlib.import_module('numpy')
            self.ts = importlib.import_module('radio.tools.pll._core_cuda')
        else:
            self.xp = importlib.import_module('numpy')
            self.ts = importlib.import_module('radio.tools.pll._core')
            self.np = self.xp

    def wave(self, freq, phase, times):
        return self.xp.cos(2.0*self.xp.pi*freq*times+phase)

    def step(self, x):
        # Make Reference Phase
        if self.times is None:
            self.times = self.xp.arange(0, 1, 1/self.fs)[:len(x)]

        # Estimate Signal Frequency
        freq = self.ts.frequency(x, self.fs)
        self.f_cb.append(freq)
        self.freq = self.np.mean(self.f_cb)

        # Get Phase Compensation
        error_ls = self.ts.alignment(x, self.algn, self.freq, self.times)
        self.phi = self.algn[self.xp.argmax(error_ls)]

    def mult(self, mult=1):
        omega = ((self.phi*mult)+(self.xp.pi/mult))
        return self.wave(self.freq*mult, omega, self.times)
