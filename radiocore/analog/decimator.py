import importlib


class Decimator:

    def __init__(self, ifs, ofs, cuda=False):
        # Import Dynamic Modules
        self.load_modules(cuda)

        # Variables to Self
        self.out_size = -1
        self.dec = ifs/ofs

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
            out = self.xs.resample_poly(out, 1, int(self.dec), window='hamm')

        if self.out_size == -1:
            self.out_size = int(len(buff)/self.dec)

        return self.xs.resample(out, self.out_size, window='hamm')
