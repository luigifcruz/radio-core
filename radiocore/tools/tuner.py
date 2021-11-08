"""Defines a Tuner module."""

from dataclasses import dataclass

from radiocore._internal import Injector


@dataclass
class Channel:
    """
    The Channel class holds frequency boundaries and other related data.

    Parameters
    ----------
    lower_frequency : float
        lower frequency boundary of the channel
    center_frequency : float
        center frequency of the channel
    higher_frequency : float
        higher frequency boundary of the channel
    bandwidth : float
        bandwidth of the channel
    """

    lower_frequency: float = 0.0
    center_frequency: float = 0.0
    higher_frequency: float = 0.0
    bandwidth: float = 0.0


class Tuner(Injector):
    """
    The Tuner class channelizes the input data into channels.

    The operation of this class assumes that the input signal
    is arranged in one second chunks. This class is based on a
    FFT, a resampler, and a IFFT. It's quite fast in the GPU.

    Parameters
    ----------
    cuda : bool
        use the GPU for processing  (default is False)
    """

    def __init__(self, cuda: bool = False):
        """Initialize the Tuner class."""
        self._cuda = cuda
        super().__init__(self._cuda)

        self._buffer = None
        self._input_frequency: int = 0.0
        self._input_bandwidth: int = 0.0
        self._bounds: Channel = []

    @property
    def input_frequency(self) -> float:
        """Return the center frequency of the input data."""
        return self._input_frequency

    @property
    def input_bandwidth(self) -> float:
        """Return the bandwidth of the input data."""
        return self._input_bandwidth

    def request_bandwidth(self, bandwidth: float):
        """
        Override the calculated bandwidth.

        The desired bandwidth should be greater than the original. The
        value set by this method will be overridden if add_channel is
        called afterward.

        Parameters
        ----------
        bandwidth : float
            desired bandwidth
        """
        if bandwidth >= self._input_bandwidth:
            self._input_bandwidth = bandwidth
        raise ValueError("requested bandwidth is too low, minimum "
                         f"is {self._input_bandwidth}")

    def add_channel(self, frequency: float, bandwidth: float):
        """
        Register a new channel to be processed.

        This call recalculates all parameters.

        Parameters
        ----------
        frequency : float
            output channel center frequency
        bandwidth : float
            output channel bandwidth
        """
        self._bounds.append(Channel(
            lower_frequency=(frequency - (bandwidth / 2)),
            center_frequency=frequency,
            higher_frequency=(frequency + (bandwidth / 2)),
            bandwidth=bandwidth
        ))
        self.__recalculate()

    def reset(self):
        """Reset the state of the Tuner."""
        self._bounds = []
        self.__recalculate()

    def load(self, input_signal):
        """
        Pre-process the input data.

        This method should be called in advance of run().

        Parameters
        ----------
        input_signal : arr
            input signal buffer with one second worth of samples
        """
        _tmp = self._xp.asarray(input_signal)
        self._buffer = self._xp.fft.fft(_tmp)

    def run(self, channel_index: int):
        """
        Return the channelized signal.

        This method should be called after load().

        Parameters
        ----------
        channel_index : int
            index of the channel
        """
        _channel = self._bounds[channel_index]
        _roll_factor = self._input_frequency - _channel.center_frequency
        _resample_factor = int(_channel.bandwidth)

        _tmp = self._xp.roll(self._buffer, _roll_factor)
        return self._xs.resample(_tmp, _resample_factor,
                                 window="hann", domain="freq")

    def __recalculate(self):
        _lower_freq = min([_ch.lower_frequency for _ch in self._bounds])
        _higher_freq = max([_ch.higher_frequency for _ch in self._bounds])

        self._input_frequency = (_lower_freq + _higher_freq) / 2
        self._input_bandwidth = (_higher_freq - _lower_freq)

        # Pad input bandwidth to make it divisable by mean channel bandwidth.
        _mean_bandwidth = sum([_ch.bandwidth for _ch in self._bounds])
        _mean_bandwidth //= len(self._bounds)

        self._input_bandwidth += (self._input_bandwidth * -1) % _mean_bandwidth
