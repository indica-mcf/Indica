"""Defines an ArrayLike type, using either the new features available
to NumPy (still bleading edge) or a rough approximation.

"""

try:
    from numpy.typing import ArrayLike
except ImportError:
    import typing
    import numpy as np
    import numbers

    ArrayLike = typing.Union[
        numbers.Number, typing.Sequence[numbers.Number], np.ndarray
    ]
