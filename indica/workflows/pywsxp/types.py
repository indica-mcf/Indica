from typing import Any

from numpy.typing import NDArray
from xarray import DataArray

from indica.workflows.pywsxp.diagnostic import Diagnostic

Config = dict[str, Any]
Diagnostics = dict[str, Diagnostic]
Inputs = dict[str, DataArray]
Results = NDArray
History = dict[str, list[Any]]
