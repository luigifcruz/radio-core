import collections
import importlib

# Source: https://gist.github.com/endolith/255291

class FFT():


    def __init__(self, fs, cuda=False):
        self.load_modules(cuda)
        self.fs = fs


    def load_modules(self, cuda):
        self.cuda = cuda

        if self.cuda:
            self.xp = importlib.import_module('cupy')
            self.np = importlib.import_module('numpy')
        else:
            self.xp = importlib.import_module('numpy')
            self.np = self.xp


    @staticmethod
    def __parabolic(f, x):
        xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
        yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
        return (xv, yv)


    def estimate(self, sig):
        f = self.xp.fft.rfft(sig) # TODO: Add window (?)
        a = self.xp.abs(f)
        i = self.xp.argmax(a)
        true_i = self.__parabolic(self.xp.log(a), i)[0]
        return self.fs * true_i / (len(f) * 2)
