"""Defines a MFM module."""

from radiocore._internal import Injector


class MFM(Injector):

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

        # Make De-emphasis Filter
        self.db, self.da = self.deemp()
        self.zi = self._xs.lfilter_zi(self.db, self.da)

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

    def run(self, buff):
        b = self._xp.array(buff)
        b = self._xp.angle(b)
        b = self._xp.unwrap(b)
        b = self._xp.diff(b)
        b = self._xp.concatenate((b, self._xp.array([b[-1]])))
        b /= self._xp.pi

        # Demod Left + Right (LPR)
        LPR, self.zi = self._xs.lfilter(self.db, self.da, b, zi=self.zi)
        LPR = self._xs.decimate(LPR, self.dec, zero_phase=True)

        # Ensure Bounds
        LPR = self._xp.clip(LPR, -0.999, 0.999)

        if self._cuda:
            return self._xp.asnumpy(LPR)

        return LPR
