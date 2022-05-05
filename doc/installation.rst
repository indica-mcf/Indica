Installation
============

InDicA requires python version 3.8 or 3.9. If this is not the default in your
environment and you cannot install either version system-wide, consider using
`pyenv <https://github.com/pyenv/pyenv>`_, available via the `installer
<https://github.com/pyenv/pyenv-installer>`_, to manage your python versions.

InDicA can be installed in a virtual environment:

.. code-block:: bash

   git clone --depth=1 https://github.com/ukaea/Indica.git analysis
   cd analysis
   python -m venv .venv
   source .venv/bin/activate
   pip install poetry
   poetry install --no-dev

You should then be able to :code:`import indica` in your python scripts,
Jupyter notebooks and python interpreters after having sourced the virtual
environment.
