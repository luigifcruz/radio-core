import collections
import importlib

from radio.tools.pll import PLL

class WBFM:

    def __init__(self, tau, ifs, ofs, cuda=False):
        # Import Dynamic Modules
        self.load_modules(cuda)

        # Variables to Self
        self.tau = tau
        self.ifs = ifs
        self.ofs = ofs
        self.dec = int(self.ifs/self.ofs)

        # Check Parameters
        assert (self.ifs % self.ofs) == 0

        # Setup Pilot PLL
        self.pll = PLL(self.ifs, cuda=self.cuda)
        self.freq = self.pll.freq

        # Setup Filters
        afb, afa = self.deemp()
        bfb, bfa = self.bandpass(23e3, 53e3)
        pfb, pfa = self.bandpass(19e3-100, 19e3+100)
        
        self.fi = {
            "afb": afb, "afa": afa,
            "bfb": bfb, "bfa": bfa,
            "pfb": pfb, "pfa": pfa,
        }

        # Setup Filters Initial Conditions
        self.zi = {
            "afr": self.xs.lfilter_zi(afb, afa),
            "afl": self.xs.lfilter_zi(afb, afa),
        }

        # Setup continuity data
        self.co = {
            "dc": collections.deque(maxlen=32),
        }

    def nyq(self, freq_hz):
        return (freq_hz / (0.5 * self.ifs))

    def deemp(self):
        lo = self.nyq(1.0 / (2 * self.np.pi * self.tau))
        hi = self.nyq(15e3)
        co = self.nyq(self.ifs/2)
        ro = self.nyq(100)

        octaves = self.np.log(hi / lo) / self.np.log(2)
        att = self.np.power(10, -((6.0 * octaves)/10))

        bounds = [0.0, lo, hi, hi+ro, co]
        gain   = [1.0, 1.0, att, 0.0, 0.0]

        taps = self.xs.firwin2(65, bounds, gain, window="hann")
        return (taps, 1.0)

    def bandpass(self, lowcut, highcut):
        lo = self.nyq(lowcut)
        hi = self.nyq(highcut)
        tp = self.xs.firwin(33, [lo, hi], pass_zero=False, window="hann")
        return (tp, 1.0)

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
        P = self.xs.filtfilt(self.fi['pfb'], self.fi['pfa'], b)
        self.pll.step(P)

        # Demod Left + Right (LPR)
        LPR, self.zi['afl'] = self.xs.lfilter(self.fi['afb'], self.fi['afa'], b, zi=self.zi['afl'])
        LPR = self.xs.decimate(LPR, self.dec, zero_phase=True) 

        # Demod Left - Right (LMR)
        LMR = self.xs.filtfilt(self.fi['bfb'], self.fi['bfa'], b)
        LMR = (self.pll.mult(2) * LMR) * 0.0175
        LMR, self.zi['afr'] = self.xs.lfilter(self.fi['afb'], self.fi['afa'], LMR, zi=self.zi['afr'])
        LMR = self.xs.decimate(LMR, self.dec, zero_phase=True)

        # Mix L+R and L-R to generate L and R
        L = LPR + LMR
        R = LPR - LMR

        # Ensure Bounds
        L = self.xp.clip(L, -0.999, 0.999)
        R = self.xp.clip(R, -0.999, 0.999)

        if self.cuda:
            L = self.xp.asnumpy(L)
            R = self.xp.asnumpy(R)

        return L, R
