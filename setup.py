from setuptools import setup, find_packages
import radio

shortdesc = "Python based library for FM, AM and WBFM demodulation."

setup(
    name='radio-core',
    version=radio.__version__,
    description=shortdesc,
    url='https://github.com/luigifreitas/radio-core',
    packages=find_packages(),
    author='Luigi F. Cruz',
    include_package_data=True
)
