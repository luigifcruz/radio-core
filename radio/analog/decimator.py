import importlib


class Decimator:

    def __init__(self, ifs, ofs, osize, cuda=False):
        # Import Dynamic Modules
        self.load_modules(cuda)

        # Variables to Self
        self.osize = osize
        self.ifs = ifs
        self.ofs = ofs
        self.dec = int(self.ifs/self.ofs)

        print("[DECIMATOR] Factor: {}".format(self.dec))

    def load_modules(self, cuda):
        self.cuda = cuda

        if self.cuda:
            self.xs = importlib.import_module('cusignal')
            self.xp = importlib.import_module('cupy')
            self.np = importlib.import_module('numpy')
            self.ss = importlib.import_module('scipy.signal')
        else:
            self.xs = importlib.import_module('scipy.signal')
            self.xp = importlib.import_module('numpy')
            self.np = self.xp
            self.ss = self.xs

    def run(self, buff):
        out = self.xp.array(buff)

        if self.dec > 1:
            out = self.xs.resample_poly(out, 1, self.dec, window='hamm')

        return self.xs.resample(out, self.osize, window='hamm')
