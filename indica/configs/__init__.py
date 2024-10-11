__all__ = []

try:
    from .readers.st40conf import ST40Conf

    __all__ += ["ST40Conf"]
except ImportError as e:
    print(e)
    pass
