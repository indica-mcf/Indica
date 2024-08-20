# InDiCA

[![Documentation
Status](https://readthedocs.org/projects/indica-ukaea/badge/?version=latest)](https://indica-ukaea.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/ukaea/Indica/workflows/tests/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Atests)
[![linting](https://github.com/ukaea/Indica/workflows/linting/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Alinting)
[![codecov](https://codecov.io/gh/ukaea/Indica/branch/master/graph/badge.svg?token=6VJ5J4JRA2)](https://codecov.io/gh/ukaea/Indica)

InDiCA (TEST) (**In**tegrated **Di**agnosti**C** **A**nalysis) is a tool which allows to perform and combine the analysis of a large number of different diagnostics from Magnetic Confined Fusion (MCF) plasma devices. It will consist of a library of functions to analyse different diagnostic systems under a common framework, and a set of workflows to combine these measurements.  <!--- Test for pre-commit -->

The overall design work has been completed and the general functionality implemented, but the library is still under active development.
<!--In  addition to the (rapidly changing) code, this repository holds the documentation for this project, [which can be found on ReadTheDocs](https://indica-ukaea.readthedocs.io/en/latest/), which is also still (rapidly) changing.-->

## Creation of development environment
1. Upgrade pip:
    - *pip install --upgrade pip*

2. Create a virtual environment with python 3.11 and activate it, e.g. using conda:
   - *set_global_conda*
   - *conda create --name ENV_NAME python=3.11*
   - *conda activate ENV_NAME*

3. Install poetry and environment dependencies:
    - *pip install poetry==1.7*
    - *poetry install*

4. For mdsplus building and installation, e.g. from a local mdsplus directory:
   - *cp -r /usr/local/mdsplus/mdsobjects/python mdsPython*
   - *cd mdsPython*
   - *python setup.py build*
   - *python setup.py install*
   - *cd ../*
   - *rm -r mdsPython*

## License

InDiCA is distributed under the [GNU General Public License version
3](LICENSE.md) or, at your option, any later version.
