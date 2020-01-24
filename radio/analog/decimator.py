import importlib


class Decimator:

    def __init__(self, ifs, ofs, osize, off=100, cuda=False):
        # Import Dynamic Modules
        self.load_modules(cuda)

        # Variables to Self
        self.osize = osize
        self.ifs = ifs
        self.ofs = ofs
        self.off = off

        self.find_ratio()

        print("[DECIMATOR] PDEC: {} -- INT/SDEC: {}/{} -- FIN: {}".format(self.pdec, self.int,
              self.sdec, self.real_fs()))

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

    def find_ratio(self):
        self.pdec = int(self.ifs/self.ofs)
        ifs = self.ifs/self.pdec
        for i in range(1, 55):
            for d in range(1, 55):
                fin = (ifs*i)/d
                if fin <= self.ofs+self.off and fin >= self.ofs:
                    self.int = i
                    self.sdec = d
                    break

    def real_fs(self):
        return (int(self.ifs/self.pdec)*self.int)/self.sdec

    def run(self, buff):
        out = self.xp.array(buff)

        if self.pdec > 1:
            out = self.xs.decimate(out, self.pdec, ftype='fir')

        if self.int > 1 or self.sdec > 1:
            out = self.xs.resample_poly(out, self.int, self.sdec, window='hamm')

        if self.osize != len(out):
            out = self.xs.resample(out, self.osize, window='hamm')

        return out
