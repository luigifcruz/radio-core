import numpy as np
from timeit import timeit
from radiocore import WBFM, MFM, FM, Decimate, Buffer, Tuner
import warnings

N_ITER = 50

class FmBenchmark:

    def __init__(self, input_size, output_size, cuda):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.cuda = cuda
        self.iterations = N_ITER

        self.wbfm = WBFM(self.input_size, self.output_size, cuda=cuda)
        self.mfm = MFM(self.input_size, self.output_size, cuda=cuda)
        self.fm = FM(self.input_size, self.output_size, cuda=cuda)
        self.buff = Buffer(self.input_size, dtype=np.complex64, cuda=cuda)

    def eval(self, func, name):
        time = timeit(func, globals={'self': self, 'np': np}, number=self.iterations)
        print('     {} scored: {}'.format(name, time / self.iterations))

    def test(self):
        print('#### FM Benchmark (Input size: {}, Output size: {}, CUDA: {}):'
              .format(self.input_size, self.output_size, self.cuda))
        self.eval('self.wbfm.run(self.buff.data)', 'WBFM')
        self.eval('self.mfm.run(self.buff.data)', 'MFM')
        self.eval('self.fm.run(self.buff.data)', 'FM')


class DecimateBenchmark:

    def __init__(self, input_size, output_size, cuda):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.cuda = cuda
        self.iterations = N_ITER

        self.decim = Decimate(self.input_size, self.output_size, cuda=cuda)
        self.buff = Buffer(self.input_size, dtype=np.complex64, cuda=cuda)

    def eval(self, func, name):
        time = timeit(func, globals={'self': self, 'np': np}, number=self.iterations)
        print('     {} scored: {}'.format(name, time / self.iterations))

    def test(self):
        print('#### Decimate Benchmark (Input size: {}, Output size: {}, CUDA: {}):'
              .format(self.input_size, self.output_size, self.cuda))
        self.eval('self.decim.run(self.buff.data)', 'Decimate')


class TunerBenchmark:

    def __init__(self, input_size, channel_size, cuda):
        self.input_size = int(input_size)
        self.channel_size = int(channel_size)
        self.cuda = cuda
        self.iterations = N_ITER

        self.tuner = Tuner(cuda=self.cuda)
        self.tuner.add_channel(94.5e6, self.channel_size, FM)
        self.tuner.add_channel(97.5e6, self.channel_size, FM)
        self.tuner.add_channel(96.9e6, self.channel_size, FM)
        self.tuner.request_bandwidth(self.input_size)

        self.buff = Buffer(self.input_size, dtype=np.complex64, cuda=cuda)

    def eval(self, func, name):
        time = timeit(func, globals={'self': self, 'np': np}, number=self.iterations)
        print('     {} scored: {}'.format(name, time / self.iterations))

    def test(self):
        print('#### Tuner Benchmark (Input size: {}, Channel size: {}, CUDA: {}):'
              .format(self.input_size, self.channel_size, self.cuda))
        self.eval('self.tuner.load(self.buff.data);self.tuner.run(0);', 'Tuner')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    FmBenchmark(256e3, 32e3, False).test()
    DecimateBenchmark(10e6, 250e3, False).test()
    DecimateBenchmark(2.5e6, 250e3, False).test()
    TunerBenchmark(10e6, 250e3, False).test()

    FmBenchmark(256e3, 32e3, True).test()
    DecimateBenchmark(10e6, 250e3, True).test()
    DecimateBenchmark(2.5e6, 250e3, True).test()
    TunerBenchmark(10e6, 250e3, True).test()
