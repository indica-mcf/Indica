"""Set up some pre-determined settings for Hypothesis.

"""

import os

from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import Verbosity

# Turn off deadlines when on CI, as that machine can be slower than my
# development machine
settings.register_profile(
    "CI", deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "debug",
    max_examples=10,
    report_multiple_bugs=False,
    verbosity=Verbosity.verbose,
)
settings.register_profile("dev", max_examples=10)

if os.getenv("CI", "false") == "true":
    settings.load_profile("CI")
