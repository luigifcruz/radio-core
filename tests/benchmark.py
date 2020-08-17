from timeit import timeit
from radio.analog import WBFM, MFM
import warnings
import numpy as np


class AnalogTest:
    
    def __init__(self, sfs, afs, mult, cuda):
        super(AnalogTest, self).__init__()
        self.sfs = int(sfs)
        self.afs = int(afs)
        self.tau = 75e-6
        self.mult = int(mult)
        self.cuda = cuda
        self.number = 500

        self.sdr_buff = 1024
        self.dsp_buff = self.sdr_buff * self.mult

        self.wbfm = WBFM(self.tau, self.sfs, self.afs, cuda)
        self.mfm = MFM(self.tau, self.sfs, self.afs, cuda)

        if self.cuda:
            import cusignal as sig
            self.buff = sig.get_shared_mem(self.dsp_buff, dtype=np.complex64)
        else:
            self.buff = np.zeros([self.dsp_buff], dtype=np.complex64)

    def eval(self, func, name):
        time = timeit(func, globals={'self': self, 'np': np}, number=self.number) / self.number
        allowance = self.dsp_buff/self.sfs
        passed = True if allowance > time else False
        print('     {} scored: {} {}({})'.format(name, time, passed, allowance))

    def test(self):
        print('#### Analog Benchmark (IFS: {}, OFS: {}, MULT: {}, CUDA: {}):'
              .format(self.sfs, self.afs, self.mult, self.cuda))
        self.eval('self.wbfm.run(self.buff)', 'WBFM')
        self.eval('self.mfm.run(self.buff)', 'MFM')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    AnalogTest(256e3, 32e3, 16, False).test()
    AnalogTest(256e3, 32e3, 16, True).test()
    AnalogTest(256e3, 32e3, 16, True).test()