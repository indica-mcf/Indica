# InDiCA

[![Documentation
Status](https://readthedocs.org/projects/indica-ukaea/badge/?version=latest)](https://indica-ukaea.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/ukaea/Indica/workflows/tests/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Atests)
[![linting](https://github.com/ukaea/Indica/workflows/linting/badge.svg)](https://github.com/ukaea/Indica/actions?query=workflow%3Alinting)
[![codecov](https://codecov.io/gh/ukaea/Indica/branch/master/graph/badge.svg?token=6VJ5J4JRA2)](https://codecov.io/gh/ukaea/Indica)

InDiCA (**In**tegrated **Di**agnosti**C** **A**nalysis) is a tool which allows to read, analyse, and model data from a wide variety of diagnostics for Magnetic Confined Fusion (MCF) devices. Combined under a common framework, diagnostic models and experimental data can be seemlessly compared, analysis and inference workflows can be implemented and tested on real as well as synthetic data. Reading functionallity is currently implemented for JET and ST40 tokamaks.

## Development environment using [uv](https://docs.astral.sh/uv/)
- The recommended UV-managed environment is CPython 3.12.
- Create or refresh it with `uv venv --python 3.12 .venv`.
- Run `uv sync --frozen --python 3.12` to let `uv` install InDiCA and the development dependencies into that environment.
- Add optional dependencies with `uv sync --extra <extraname> --frozen --python 3.12` or `uv sync --all-extras --frozen --python 3.12`.
- Adding using `uv add` keeps the `uv.lock` file up-to-date automatically.

### [Aurora](https://github.com/fsciortino/Aurora) install instructions
Currently there's a build-system issue due to `scikit_build` version, so Aurora is not automatically installed. To install:
1. `uv sync --extra aurora --frozen --python 3.12` or `uv sync --all-extras --frozen --python 3.12` to get the build requirements into the UV environment
2. `uv pip install --python .venv/bin/python --no-build-isolation aurorafusion` to install the package into the active environment

## License
InDiCA is distributed under the [GNU General Public License version
3](LICENSE.md) or, at your option, any later version.
