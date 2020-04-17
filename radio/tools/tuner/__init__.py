import importlib


class Tuner:

    def __init__(self, bands, osize, cuda=False):
        # Import Dynamic Modules
        self.load_modules(cuda)

        # Variables to Self
        self.bands = bands

        # List Bands Boundaries
        los = self.xp.array([(b['freq'] - (b['bw']/2)) for b in self.bands])
        his = self.xp.array([(b['freq'] + (b['bw']/2)) for b in self.bands])

        # Get Bands Boundaries
        self.lof = self.xp.min(los)
        self.hif = self.xp.max(his)

        # Get Bandwidth and Center Frequency
        self.bw = float(self.hif-self.lof)
        self.bw += self.bands[0]['bw']-(self.bw % self.bands[0]['bw'])
        self.mdf = float((self.lof+self.hif)/2.0)

        # Get Size
        self.size = int(osize*(self.bw//self.bands[0]['bw']))

        # List Decimation Factor
        self.dfac = [int(self.size/(self.bw//b['bw'])) for b in self.bands]

        # List Frequency & FFT Offset
        self.foff = [b['freq'] - self.mdf for b in self.bands]
        self.toff = [-(self.size*f)/self.bw for f in self.foff]

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

    def load(self, buff):
        self.b = self.xp.fft.fft(self.xp.array(buff))

    def run(self, id):
        a = self.xp.roll(self.b, int(self.toff[id]))
        a = self.xs.resample(a, self.dfac[id], window='hamm', domain="freq")
        return a
