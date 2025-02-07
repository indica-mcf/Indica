"""Set up some pre-determined settings for Hypothesis.

"""

import sys
from unittest import mock

# Turn off import of modules that cnnot be installed in CI
sys.modules["indica.readers.st40reader"] = mock.MagicMock()
sys.modules["indica.readers.st40reader.ST40Reader"] = mock.MagicMock()
sys.modules["indica.writers.bda_tree"] = mock.Mock()
