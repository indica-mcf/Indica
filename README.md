# InDiCA

[![Documentation
Status](https://readthedocs.org/projects/indica-ukaea/badge/?version=latest)](https://indica-ukaea.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/ukaea/Indica/workflows/tests/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Atests)
[![linting](https://github.com/ukaea/Indica/workflows/linting/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Alinting)
[![codecov](https://codecov.io/gh/ukaea/Indica/branch/master/graph/badge.svg?token=6VJ5J4JRA2)](https://codecov.io/gh/ukaea/Indica)

InDiCA (**In**tegrated **Di**agnosti**C** **A**nalysis) is a tool which allows to perform and combine the analysis of a large number of different diagnostics from Magnetic Confined Fusion (MCF) plasma devices. It will consist of a library of functions to analyse different diagnostic systems under a common framework, and a set of workflows to combine these measurements.  <!--- Test for pre-commit -->

The overall design work has been completed and the general functionality implemented, but the library is still under active development.

## Development environment using [uv](https://docs.astral.sh/uv/)
- The recommended UV-managed environment is CPython 3.12.
- Create or refresh it with `uv venv --python 3.12 .venv`.
- Run `uv sync --frozen --python 3.12` to let `uv` install InDiCA and the development dependencies into that environment.
- Add optional dependencies with `uv sync --extra <extraname> --frozen --python 3.12` or `uv sync --all-extras --frozen --python 3.12`.

### Adding package
- To add a main dependency: `uv add "package"`
- To add an optional extra: `uv add --optional=<groupname> "package"`
- To add a development dependency: `uv add --dev "package"`

Adding using `uv add` keeps the `uv.lock` file up-to-date automatically. See [uv docs](https://docs.astral.sh/uv/concepts/projects/dependencies/) for more information.

### [Aurora](https://github.com/fsciortino/Aurora) install instructions
Currently there's a build-system issue with Aurora that fails on install due to `scikit_build` version, so the Aurora extra only installs the build requirements without the actual package. To install:
1. `uv sync --extra aurora --frozen --python 3.12` or `uv sync --all-extras --frozen --python 3.12` to get the build requirements into the UV-managed Python 3.12 environment
2. `uv pip install --python .venv/bin/python --no-build-isolation aurorafusion` to install the package into the active environment

## (LEGACY) Creation of development environment
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
