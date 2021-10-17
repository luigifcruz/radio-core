"""Defines a Chopper module."""

from radio._internal import Injector


class Tuner(Injector):
    """
    The Tuner class implements a variable bandwidth channelizer.

    Attributes
    ----------
    bands : dict
    """

    def __init__(self, bands, osize, cuda=False):
        self._cuda = cuda

        super().__init__(self._cuda)

        # Variables to Self
        self.bands = bands
        self.fft_size = -1

        # List Bands Boundaries
        los = self._xp.array([(b['freq'] - (b['bw']/2)) for b in self.bands])
        his = self._xp.array([(b['freq'] + (b['bw']/2)) for b in self.bands])

        # Get Bands Boundaries
        self.lof = self._xp.min(los)
        self.hif = self._xp.max(his)

        # Get Bandwidth and Center Frequency
        self.bwt = float(self.hif-self.lof)
        self.bwt += self.bands[0]['bw']-(self.bwt % self.bands[0]['bw'])
        self.mdf = float((self.lof+self.hif)/2.0)

        # Get Size
        self.size = int(osize*(self.bwt//self.bands[0]['bw']))

        # List Decimation Factor
        self.dfac = [int(self.size/(self.bwt//b['bw'])) for b in self.bands]

        # List Frequency & FFT Offset
        self.foff = [b['freq'] - self.mdf for b in self.bands]
        self.toff = [-(self.size*f)/self.bwt for f in self.foff]

    def load(self, a):
        a = self.xp.array(a)
        self.b = self.xfp.fft(a)

    def run(self, id):
        a = self.xp.roll(self.b, int(self.toff[id]))
        a = self.xs.resample(a, self.dfac[id], domain="freq")
        return a
