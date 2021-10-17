# 📔 radio-core
### Python based DSP for [CyberRadio](https://github.com/luigifreitas/CyberRadio) and [PyRadio](https://github.com/luigifreitas/PyRadio).
Accelerated on the GPU with CUDA by [#cuSignal](https://github.com/rapidsai/cusignal).

## Features

### Analog
- **WBFM**: Demodulation of wideband FM stations with Stereo Support. Supports 75uS and 50uS de-emphasis.
- **MFM**: Demodulation of wideband FM stations without Stereo Support. Supports 75uS and 50uS de-emphasis.

### Tools
- **PLL**: Clock-recovery and phase estimation for real signals.
- **Tuner**: Fast combo-tuning of wideband signals using FFT, IFFT, and polyphase-decimation.

## Installation
To enable GPU acceleration, make sure to install [Cupy](https://docs.cupy.dev/en/stable/install.html) and
[cuSignal](https://github.com/rapidsai/cusignal#installation) first.

```
$ python -m pip install git+https://github.com/luigifreitas/radio-core.git
```

## Hacking
If you are interested in the core DSP, you are in the right place! If you want to tinker with the code and make your own application, you should look for the [PyAudio](https://github.com/luigifreitas/PyRadio) Repository. If you just want to use to listen to some good music, try the [CyberRadio](https://github.com/luigifreitas/CyberRadio) Desktop App.
