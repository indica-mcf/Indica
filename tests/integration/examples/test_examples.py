import indica.examples as examples
from indica.examples import __all__


def test_all_examples():
    for function_name in __all__:
        _ = getattr(examples, function_name)()
