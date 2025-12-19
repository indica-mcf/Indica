from pathlib import Path
from typing import TextIO

import numpy as np

from indica.readers import ADASReader


class MockReader(ADASReader):
    test_file: Path

    def _get_file(self, *args, **kwargs) -> TextIO:
        return self.test_file.open("r")


class TestADF21ADF22:
    reader = MockReader()

    def test_read(self):
        self.reader.test_file = Path(__file__).parent / "test_adf21.dat"
        data = self.reader._get_adf21_adf22(
            dataclass="",
            beam="",
            element="",
            charge="",
            quantity="bms",
            year="",
        )
        ref = np.load(Path(__file__).parent / "test_adf21.npz")
        assert np.all(np.isclose(data.to_numpy(), ref["data"], rtol=1e-5, atol=1e-5))
