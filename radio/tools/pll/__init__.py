import collections
import importlib

class PLL:


    def __init__(self, fs, cuda=False):
        self.load_modules(cuda)
        self.freq_est = self.fe.FFT(fs, cuda)
        self.estimated_freq = 0.0
        self.reference = None
        self.baseline = None
        self.cache = None
        self.valid = 0.0
        self.phi = 0.0
        self.fs = fs
        self.len = 0


    def load_modules(self, cuda):
        self.cuda = cuda

        # Load common modules.
        self.fe = importlib.import_module('radio.tools.festimator')

        # Load runtime dependent modules.
        if self.cuda:
            self.xp = importlib.import_module('cupy')
            self.np = importlib.import_module('numpy')
        else:
            self.xp = importlib.import_module('numpy')
            self.np = self.xp


    def wave(self, mult=1.0, phi=0.0):
        return self.xp.cos((2 * self.xp.pi * self.baseline * self.estimated_freq * mult) - phi)


    def step(self, x):
        # Run post-initialization once.
        if self.baseline is None:
            self.len = len(x)
            self.valid = 0.100 / ((1 / self.fs) * self.len)
            self.baseline = self.xp.arange(0, (1 / self.fs) * self.len, 1 / self.fs)

        # Run frequency estimation every 100ms.
        if self.cache is None or self.cache > self.valid:
            self.estimated_freq = self.freq_est.estimate(x)
            self.reference = self.wave()
            self.cache = 0
        self.cache += 1

        # Calculate phase error between reference and input.
        dot = self.xp.dot(self.reference, x)
        norm = self.xp.linalg.norm(self.reference) * self.xp.linalg.norm(x)
        self.phi = self.xp.arccos(dot / norm)


    def mult(self, mult=1.0):
        return self.wave(mult, self.phi)
