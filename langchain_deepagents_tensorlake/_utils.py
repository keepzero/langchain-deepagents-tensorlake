"""Shared validation and read-bound handling for Tensorlake file backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import FileData, ReadResult
from deepagents.backends.utils import (
    compile_grep_include_glob as _compile_grep_include_glob,
)
from deepagents.backends.utils import slice_read_response as _slice_read_response

if TYPE_CHECKING:
    from collections.abc import Callable


def empty_read_result() -> ReadResult:
    """Return an empty page with no-lines metadata when the protocol supports it."""
    data = FileData(content="", encoding="utf-8")
    # Early 0.7 releases have pagination but lack the optional no-lines flag.
    if "no_lines_requested" in ReadResult.__dataclass_fields__:
        return ReadResult(file_data=data, no_lines_requested=True)
    return ReadResult(file_data=data)


def slice_read_response(file_data: FileData, offset: int, limit: int) -> ReadResult:
    """Normalize read bounds before delegating to the native pagination helper."""
    offset, limit = max(0, int(offset)), max(0, int(limit))
    if limit == 0 and file_data["content"].strip():
        return empty_read_result()
    return _slice_read_response(file_data, offset, limit)


def compile_grep_include_glob(pattern: str) -> Callable[[str], bool]:
    """Validate include patterns and normalize native glob compilation errors."""
    if ".." in pattern.replace("\\", "/").split("/"):
        message = "invalid_glob_pattern: path traversal is not allowed"
        raise ValueError(message)
    try:
        return _compile_grep_include_glob(pattern)
    except Exception as error:
        # Early 0.7 helpers expose private wcmatch expansion-limit exceptions.
        message = "invalid_glob_pattern: could not compile the include pattern"
        raise ValueError(message) from error
