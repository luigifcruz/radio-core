"""Defines a WBFM module."""

from radiocore._internal import Injector
from radiocore.analog import PLL

class WBFM(Injector):

    def __init__(self, tau, ifs, ofs, cuda=False):
        self._cuda = cuda

        super().__init__(cuda)

        # Variables to Self
        self.tau = tau
        self.ifs = ifs
        self.ofs = ofs
        self.dec = int(self.ifs/self.ofs)

        # Check Parameters
        assert (self.ifs % self.ofs) == 0

        # Setup Pilot PLL
        self.pll = PLL(cuda=self._cuda)

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
            "afr": self._xs.lfilter_zi(afb, afa),
            "afl": self._xs.lfilter_zi(afb, afa),
            "pfb": self._xs.lfilter_zi(pfb, pfa),
        }

    def nyq(self, freq_hz):
        return (freq_hz / (0.5 * self.ifs))

    def deemp(self):
        lo = self.nyq(1.0 / (2 * self._np.pi * self.tau))
        hi = self.nyq(15e3)
        co = self.nyq(self.ifs/2)
        ro = self.nyq(100)

        octaves = self._np.log(hi / lo) / self._np.log(2)
        att = self._np.power(10, -((6.0 * octaves)/10))

        bounds = [0.0, lo, hi, hi+ro, co]
        gain   = [1.0, 1.0, att, 0.0, 0.0]

        taps = self._xs.firwin2(65, bounds, gain, window="hann")
        return (taps, 1.0)

    def bandpass(self, lowcut, highcut):
        lo = self.nyq(lowcut)
        hi = self.nyq(highcut)
        tp = self._xs.firwin(33, [lo, hi], pass_zero=False, window="hann")
        return (tp, 1.0)

    def run(self, buff):
        b = self._xp.array(buff, copy=False)
        b = self._xp.angle(b)
        b = self._xp.unwrap(b)
        b = self._xp.diff(b)
        b = self._xp.concatenate((b, self._xp.array([b[-1]])))
        b /= self._xp.pi

        # Synchronize PLL with Pilot
        P, self.zi['pfb'] = self._xs.lfilter(self.fi['pfb'], self.fi['pfa'], b, zi=self.zi['pfb'])
        self.pll.step(P)

        # Demod Left + Right (LPR)
        LPR, self.zi['afl'] = self._xs.lfilter(self.fi['afb'], self.fi['afa'], b, zi=self.zi['afl'])
        LPR = self._xs.decimate(LPR, self.dec, zero_phase=True)

        # Demod Left - Right (LMR)
        LMR = self._xs.filtfilt(self.fi['bfb'], self.fi['bfa'], b)
        LMR = (self.pll.wave(2) * LMR) * 1.0175
        LMR, self.zi['afr'] = self._xs.lfilter(self.fi['afb'], self.fi['afa'], LMR, zi=self.zi['afr'])
        LMR = self._xs.decimate(LMR, self.dec, zero_phase=True)

        # Mix L+R and L-R to generate L and R
        L = LPR + LMR
        R = LPR - LMR

        # Remove DC from signal.
        L -= self._xp.mean(L)
        R -= self._xp.mean(R)

        # Ensure Bounds
        L = self._xp.clip(L, -0.999, 0.999)
        R = self._xp.clip(R, -0.999, 0.999)

        if self._cuda:
            L = self._xp.asnumpy(L)
            R = self._xp.asnumpy(R)

        return L, R
