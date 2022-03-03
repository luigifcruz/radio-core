# 📻 Radio Core
### Real-world signal processing functions for Python.

## Features
- ⚙️ Compatible with the majority of SDRs via [SoapySDR](https://github.com/pothosware/SoapySDR).
- ⚡️ Accelerated on Nvidia GPUs with CUDA via [CuPy](https://github.com/cupy/cupy/) and [cuSignal](https://github.com/rapidsai/cusignal).
- 🚀 Runs smoothly in the Raspberry Pi 4, Nvidia Jetson, and Apple Silicon.

## Functions

### Analog
- **PLL**: Clock-recovery and phase estimation for real-valued signals.
- **WBFM**: Demodulation of wideband FM stations with Stereo Support. Supports de-emphasis.
- **MFM**: Demodulation of wideband FM stations without Stereo Support. Supports de-emphasis.
- **FM**: Demodulation of FM transmissions.
- **Deemphasis**: De-emphasize audio.
- **Decimate**: Resample signal.
- **Bandpass**: Filter signal with bandpass window.

### Tools
- **Tuner**: Channelize the input data into smaller channels.
- **Ringbuffer**: Zero-copy variable length circular buffer implemented in Python.
- **Carrousel**: Zero-copy fixed length circular buffer implemented in Python.
- **Chopper**: Divide a larger array into smaller fixed side elements.
- **Buffer**: Provide an array allocated in the GPU or CPU.

## Examples

- **Receive FM**: Receive and play single wideband FM radio station.
- **Multi FM**: Receive multiple wideband FM radio stations. Broadcast audio via ZeroMQ.

## Installation

### System Dependencies
- SoapySDR Base ([Repo](https://github.com/pothosware/SoapySDR))
- SoapySDR Modules ([LimeSuite](https://github.com/myriadrf/LimeSuite.git), [AirSpyOne](https://github.com/pothosware/SoapyAirspy), [AirSpyHF](https://github.com/pothosware/SoapyAirspyHF), [PlutoSDR](https://github.com/pothosware/SoapyPlutoSDR), [RTL-SDR](https://github.com/pothosware/SoapyRTLSDR))
- Python 3.7+
- PulseAudio

#### Ubuntu/Debian
After installing the base SoapySDR and its modules, install the direct dependencies with `apt`:
```bash
$ apt install libpulse-dev libsamplerate-dev libasound2-dev portaudio19-dev
```

### Python

#### CPU
```
$ python -m pip install git+https://github.com/luigifcruz/radio-core.git
```
#### GPU (CUDA)
```
$ python -m pip install "git+https://github.com/luigifcruz/radio-core.git#egg=radiocore[cuda]"
```

## Validated Radios
- AirSpy HF+ Discovery
- AirSpy R2
- LimeSDR Mini/USB
- PlutoSDR
- RTL-SDR
- SDRPlay

## Hacking
If you are interested in the core DSP, you are in the right place! Feel free to tinker with the code and make your own application. If you just want to use to listen to some good music, try the [CyberRadio](https://github.com/luigifreitas/CyberRadio) Desktop App.
