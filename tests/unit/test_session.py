"""Check PROV/session tracking functionality."""

from hypothesis.strategies import composite
from hypothesis.strategies import emails

from src.session import Session


@composite
def sessions(draw):
    """Generates :py:class:`src.session.Session` objects."""
    # TODO: add option to draw ORCID identifiers
    return Session(draw(emails()))


# Test hashing returns unique values

# Check session initialises PROV properly

# Check session behaves properly as context manager

# Try pushing/popping/accessing PROV agents (RuleBasedStateMachine)

# Test begin() factory method

# Test exporting/reloading data produces same result

# Test export/reload properly interacts with data in __main__
