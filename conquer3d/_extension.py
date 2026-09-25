"""Loads the compiled ``conquer3d._C`` extension, or stands in for it when it is absent.

Every operator in the library lives in the CUDA extension, and importing it at module
scope meant the whole package needed a matching GPU build before even its names could be
read. That is fine on a workstation and awkward everywhere else -- type checking, docs
tooling and CI all want to look at the API without compiling it.

So the import is attempted once here. When it succeeds nothing else happens and
``conquer3d._C`` is the real module. When it fails a stand-in is registered under the same
name, which lets ``from .._C import marching_cubes`` and friends keep working at import
time; the error surfaces when something is actually called, and carries the original
failure with it.
"""

import importlib
import importlib.machinery
import sys
import types
from typing import Any

__all__ = ["HAS_EXTENSION", "ExtensionUnavailableError", "extension_error"]

_MODULE = "conquer3d._C"

_HINT = """\
Build it in place with

    TORCH_CUDA_ARCH_LIST=<your arch> python setup.py build_ext --inplace

or install a prebuilt wheel with `pip install -U conquer3d`."""


class ExtensionUnavailableError(ImportError):
    """Raised when something needs the CUDA extension and it could not be loaded.

    Subclasses :class:`ImportError`, since the cause is always a module that failed to
    import; it is raised at the point of use rather than at import so that the package
    can be inspected without a GPU build.
    """


def _fail(name: str) -> "ExtensionUnavailableError":
    """Builds the error raised when `name` is used without the extension."""
    cause = f"\n\nThe original import failed with:\n    {type(_ERROR).__name__}: {_ERROR}" if _ERROR else ""
    return ExtensionUnavailableError(
        f"conquer3d.{name} needs the compiled CUDA extension ({_MODULE}), which is not "
        f"available. The rest of the package still imports so the API can be read without "
        f"a GPU build.{cause}\n\n{_HINT}"
    )


def _placeholder(name: str) -> type:
    """A stand-in for one extension symbol: importable, and informative when used."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _fail(name)

    return type(
        name,
        (object,),
        {
            "__init__": __init__,
            # Report the module the symbol would have come from, so a repr or a traceback
            # points at the extension rather than at this shim.
            "__module__": _MODULE,
            "__doc__": f"Unavailable stand-in for {_MODULE}.{name}.",
        },
    )


class _MissingExtension(types.ModuleType):
    """Stands in for ``conquer3d._C`` when the extension is not built.

    Any attribute resolves to a placeholder class, so the ``from .._C import X`` lines
    scattered through the package still bind a name. Calling or constructing that name
    raises :class:`ExtensionUnavailableError`.
    """

    def __getattr__(self, name: str) -> type:
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = _placeholder(name)
        setattr(self, name, placeholder)
        return placeholder


try:
    _C = importlib.import_module(_MODULE)
    _ERROR: Any = None
    #: Whether the compiled CUDA extension loaded. False means the package imports but
    #: every operator raises :class:`ExtensionUnavailableError` when called.
    HAS_EXTENSION = True
except Exception as exc:  # noqa: BLE001 - a broken build fails in many ways, all equivalent here
    _ERROR = exc
    _C = _MissingExtension(_MODULE)
    # A spec keeps the stand-in indistinguishable from a real module to the import system,
    # which otherwise warns when a `sys.modules` entry has none.
    _C.__spec__ = importlib.machinery.ModuleSpec(_MODULE, loader=None)
    sys.modules[_MODULE] = _C
    HAS_EXTENSION = False


def extension_error() -> Any:
    """Returns the exception that stopped the extension loading, or None if it loaded.

    Returns:
        Optional[BaseException]: The original import failure, kept so that a broken build
        is distinguishable from an absent one.

    Example:
        >>> import conquer3d
        >>> if not conquer3d.HAS_EXTENSION:
        ...     print(conquer3d.extension_error())
    """
    return _ERROR
