class PolyxiosError(Exception):
    """Base exception for all polyxios errors."""


class ValidationError(PolyxiosError):
    """Raised when a PolyData fails structural validation."""


class CodecError(PolyxiosError):
    """Raised for codec-level read/write errors."""


class UnsupportedFormatError(PolyxiosError):
    """Raised when no codec is registered for a file format."""


class MissingPackageError(UnsupportedFormatError, AttributeError):
    """Raised when a codec reaches for an optional package that is not installed.

    An :class:`UnsupportedFormatError`, because a format whose reader is not
    there is one polyxios cannot read here, and the caller's ``except`` for
    the one catches the other. An :class:`AttributeError` too, so that
    ``hasattr`` and ``getattr(..., default)`` on the stand-in a missing
    package leaves behind answer the way they do on anything else.
    """


class LazyReadError(PolyxiosError):
    """Raised when lazy=True is requested for a format that does not support it."""


class FetcherError(PolyxiosError):
    """Raised when asset resolution or retrieval fails."""


class IndexOverflowError(CodecError):
    """
    Raised when connectivity.max() exceeds a codec's MAX_CONNECTIVITY_INDEX at write time.
    Never silently downcast int64 to int32.
    """

    def __init__(self, codec: str, max_allowed: int, actual_max: int) -> None:
        self.codec = codec
        self.max_allowed = max_allowed
        self.actual_max = actual_max
        super().__init__(
            f"Codec '{codec}': connectivity index {actual_max} exceeds maximum "
            f"{max_allowed}. Use a format that supports int64 connectivity."
        )


class UnknownElementTypeError(UnsupportedFormatError):
    """
    Raised when a file contains an element type code not in _element_types.py.
    Never raise IndexError or KeyError for unknown element types.
    """

    def __init__(self, fmt: str, type_code: int) -> None:
        self.fmt = fmt
        self.type_code = type_code
        super().__init__(f"Format '{fmt}': unknown element type code {type_code}.")
