# InDiCA

[![Documentation
Status](https://readthedocs.org/projects/indica-ukaea/badge/?version=latest)](https://indica-ukaea.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/ukaea/Indica/workflows/tests/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Atests)
[![linting](https://github.com/ukaea/Indica/workflows/linting/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Alinting)
[![codecov](https://codecov.io/gh/ukaea/Indica/branch/master/graph/badge.svg?token=6VJ5J4JRA2)](https://codecov.io/gh/ukaea/Indica)

InDiCA (**In**tegrated **Di**agnosti**C** **A**nalysis) is a tool which allows to perform and combine the analysis of a large number of different diagnostics from Magnetic Confined Fusion (MCF) plasma devices. It will consist of a library of functions to analyse different diagnostic systems under a common framework, and a set of workflows to combine these measurements.  <!--- Test for pre-commit -->

Currently under active development are workflows for the calculation of the plasma composition following the methodologies explained in [M. Sertoli et al., J. Plasma Phys. (2019), vol. 85, 905850504](https://doi.org/10.1017/S0022377819000618), and for constrining the shape of the kinetic profiles given LOS and volume integrated measurements only. Diagnostic forward models presently under development are: SXR and bolometer cameras, passive spectrometers, interferometers and magnetic measurements.

The overall design work has been completed and the general functionality implemented, but the library is still under active development. In  addition to the (rapidly changing) code, this repository holds the documentation for this project, [which can be found on
ReadTheDocs](https://indica-ukaea.readthedocs.io/en/latest/), which is also still (rapidly) changing.

## Install
1. Make sure pip is up-to-date:
    - *pip install --upgrade pip*

2. Create a virtual environment with python 3.9 and activate it.

3. Install poetry and environment dependencies:
    - *pip install poetry==1.1.15*
    - *poetry update*

4. For mdsplus building and installation:
   - *cp -r /usr/local/mdsplus/mdsobjects/python mdsPython*
   - *cd mdsPython*
   - *python setup.py build*
   - *python setup.py install*
   - *cd ../*
   - *rm -r mdsPython*



## License

InDiCA is distributed under the [GNU General Public License version
3](LICENSE.md) or, at your option, any later version.
