"""Defines a Decimator module."""

from radiocore._internal import Injector


class Decimator(Injector):

    def __init__(self, ifs, ofs, cuda=False):
        self._cuda = cuda

        super().__init__(self._cuda)

        # Variables to Self
        self.out_size = -1
        self.dec = ifs/ofs

        print("[DECIMATOR] Factor: {}".format(self.dec))

    def run(self, buff):
        out = self._xp.array(buff)

        if self.dec > 1:
            out = self._xs.resample_poly(out, 1, int(self.dec), window='hamm')

        if self.out_size == -1:
            self.out_size = int(len(buff)/self.dec)

        return self._xs.resample(out, self.out_size, window='hamm')
