import importlib


class MFM:

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

        # Make De-emphasis Filter
        self.db, self.da = self.deemp()
        self.zi = self.xs.lfilter_zi(self.db, self.da)

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

    def run(self, buff):
        b = self.xp.array(buff)
        b = self.xp.angle(b)
        b = self.xp.unwrap(b)
        b = self.xp.diff(b)
        b = self.xp.concatenate((b, self.xp.array([b[-1]])))
        b /= self.xp.pi

        # Demod Left + Right (LPR)
        LPR, self.zi = self.xs.lfilter(self.db, self.da, b, zi=self.zi)
        LPR = self.xs.decimate(LPR, self.dec, zero_phase=True)

        # Ensure Bounds
        LPR = self.xp.clip(LPR, -0.999, 0.999)

        if self.cuda:
            return self.xp.asnumpy(LPR)

        return LPR
