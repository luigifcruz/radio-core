import collections
import importlib

from radio.tools.pll import PLL
from radio.tools.helpers import lfilter, filtfilt


class WBFM:

    def __init__(self, tau, ifs, ofs, isize, cuda=False, numba=False):
        # Import Dynamic Modules
        self.load_modules(cuda, numba)

        # Variables to Self
        self.tau = tau
        self.ifs = ifs
        self.ofs = ofs
        self.dec = int(self.ifs/self.ofs)
        self.isize = isize
        self.out = self.isize//self.dec

        # Check Parameters
        assert (self.ifs % self.ofs) == 0

        # Setup Pilot PLL
        self.pll = PLL(self.ifs, self.isize, cuda=self.cuda)
        self.freq = self.pll.freq

        # Setup Filters
        x = self.np.exp(-1/(self.ofs * self.tau))
        pb, pa = self.ss.butter(2, [19e3-200, 19e3+200], btype='band', fs=self.ifs)
        mb, ma = self.ss.butter(10, 15e3, btype='low', fs=self.ifs)
        hb, ha = self.ss.butter(2, 40, btype='high', fs=self.ifs)

        self.fi = {
            "db": [1-x], "da": [1, -x],
            "pb": pb, "pa": pa,
            "mb": mb, "ma": ma,
            "hb": hb, "ha": ha,
        }

        # Setup Filters Initial Conditions
        self.zi = {
            "mlpr": self.ss.lfilter_zi(mb, ma),
            "mlmr": self.ss.lfilter_zi(mb, ma),
            "dlpr": self.ss.lfilter_zi(self.fi["db"], self.fi["da"]),
            "dlmr": self.ss.lfilter_zi(self.fi["db"], self.fi["da"]),
            "hlpr": self.ss.lfilter_zi(self.fi["hb"], self.fi["ha"]),
        }

        # Setup continuity data
        self.co = {
            "dc": collections.deque(maxlen=32),
            "diff": self.xp.array([0.0]),
        }

    def load_modules(self, cuda, numba):
        self.cuda = cuda
        self.numba = numba

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
        b = self.xp.array(buff)
        b = self.xp.angle(b)
        b = self.xp.unwrap(b)
        b = self.xp.diff(b)
        b = self.xp.concatenate((b, self.xp.array([b[-1]])))
        b /= self.xp.pi

        # Normalize for DC
        dc = self.xp.mean(b)
        self.co['dc'].append(dc)
        b -= self.np.mean(self.co['dc'])

        # Synchronize PLL with Pilot
        P = filtfilt(self, self.fi['pb'], self.fi['pa'], b)
        self.pll.step(P)

        # Demod Left + Right (LPR)
        LPR, self.zi['mlpr'] = lfilter(self, self.fi['mb'], self.fi['ma'], b, zi=self.zi['mlpr'])
        LPR, self.zi['hlpr'] = lfilter(self, self.fi['hb'], self.fi['ha'], LPR, zi=self.zi['hlpr'])
        LPR = self.xs.resample_poly(LPR, 1, self.dec, window='hamm')
        LPR, self.zi['dlpr'] = lfilter(self, self.fi['db'], self.fi['da'], LPR, zi=self.zi['dlpr'])

        # Demod Left - Right (LMR)
        LMR = (self.pll.mult(2) * b) * 1.02
        LMR, self.zi['mlmr'] = lfilter(self, self.fi['mb'], self.fi['ma'], LMR, zi=self.zi['mlmr'])
        LMR = self.xs.resample_poly(LMR, 1, self.dec, window='hamm')
        LMR, self.zi['dlmr'] = lfilter(self, self.fi['db'], self.fi['da'], LMR, zi=self.zi['dlmr'])

        # Mix L+R and L-R to generate L and R
        L = LPR+LMR
        R = LPR-LMR

        # Ensure Bounds
        L = self.xp.clip(L, -1.0, 1.0)
        R = self.xp.clip(R, -1.0, 1.0)

        if self.cuda:
            L = self.xp.asnumpy(L)
            R = self.xp.asnumpy(R)

        return L, R
