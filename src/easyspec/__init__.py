
__author__ = "Raniere de Menezes"

import warnings as _warnings

try:
    from importlib.metadata import version as _package_version
    __version__ = _package_version("easyspec")   # version of the installed package (from setup.py)
except Exception:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")   # e.g. running from a source checkout, without installing
        from .version import __version__
from ._banner import show_banner as _show_banner

_show_banner(__version__)   # CBPF logo (Jupyter) or short text (terminal); disable with the environment variable EASYSPEC_NO_BANNER=1
del _show_banner, _warnings
