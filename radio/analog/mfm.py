import collections
import importlib


class MFM:

    def __init__(self, tau, ifs, ofs, cuda=False, numba=False):
        # Import Dynamic Modules
        self.load_modules(cuda, numba)

        # Variables to Self
        self.tau = tau
        self.ifs = ifs
        self.ofs = ofs
        self.dec = int(self.ifs/self.ofs)

        # Check Parameters
        assert (self.ifs % self.ofs) == 0

        # Make De-emphasis Filter
        x = self.np.exp(-1/(self.ofs * self.tau))
        self.db = [1-x]
        self.da = [1, -x]
        self.zi = self.ss.lfilter_zi(self.db, self.da)

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
        d = self.xp.concatenate((self.co['diff'], self.xp.angle(b)), axis=None)
        b = self.xp.diff(self.xp.unwrap(d))
        self.co['diff'][0] = d[-1]
        b /= self.xp.pi

        # Normalize for DC
        dc = self.xp.mean(b)
        self.co['dc'].append(dc)
        b -= self.np.mean(self.co['dc'])

        # Demod Left + Right (LPR)
        LPR = self.xs.resample_poly(b, 1, self.dec, window='hamm')
        LPR, self.zi = self.xs.lfilter(self.db, self.da, LPR, zi=self.zi)

        # Ensure Bounds
        LPR = self.xp.clip(LPR, -1.0, 1.0)

        if self.cuda:
            return self.xp.asnumpy(LPR)

        return LPR
