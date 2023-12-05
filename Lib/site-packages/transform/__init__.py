from .mql import MQLClient  # noqa: E402

__version__ = "1.2.0"
PACKAGE_NAME = "transform"

# mql gets imported if user is already authenticated
mql = None
try:
    mql = MQLClient()
except Exception as e:  # noqa: D
    pass
