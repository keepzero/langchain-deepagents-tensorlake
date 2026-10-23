"""Deep Agents filesystem backend backed by Tensorlake Cloud Volumes."""

from __future__ import annotations

import base64
from functools import wraps
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    _get_backend_read_file_type,
    perform_string_replacement,
)
from tensorlake.filesystem import FileEntry, Filesystem
from tensorlake.filesystem.exceptions import (
    FileNotFoundInFilesystemError,
    FilesystemAPIError,
    FilesystemError,
)

from langchain_deepagents_tensorlake._utils import (
    compile_grep_include_glob,
    empty_read_result,
    slice_read_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_SEARCH_FILES = 10_000
MAX_SEARCH_BYTES = 64 * 1024 * 1024
_Params = ParamSpec("_Params")
_Result = TypeVar(
    "_Result",
    LsResult,
    ReadResult,
    WriteResult,
    EditResult,
    DeleteResult,
    GlobResult,
    GrepResult,
)


def _volume_path(path: str, *, allow_root: bool = False) -> str:
    """Translate virtual absolute paths to filesystem-relative paths."""
    parsed = PurePosixPath(path)
    if (
        not path.startswith("/")
        or path.startswith("//")
        or "\x00" in path
        or "\\" in path
        or any(part in {"..", "~"} for part in parsed.parts)
        or (not allow_root and not parsed.name)
    ):
        message = "invalid_path: use an absolute path within the filesystem"
        raise ValueError(message)
    return str(parsed).lstrip("/")


def _storage_error(error: FilesystemError) -> str:
    """Raw provider errors can contain signed URLs or credentials."""
    if isinstance(error, FileNotFoundInFilesystemError):
        return "file_not_found"
    if isinstance(error, FilesystemAPIError) and error.status_code in {401, 403}:
        return "permission_denied"
    if isinstance(error, FilesystemAPIError):
        return f"storage_unavailable: {type(error).__name__}, HTTP {error.status_code}"
    return f"storage_unavailable: {type(error).__name__}"


def _file_result(
    result_type: type[_Result],
) -> Callable[
    [Callable[Concatenate[TensorlakeFilesystemBackend, _Params], _Result]],
    Callable[Concatenate[TensorlakeFilesystemBackend, _Params], _Result],
]:
    """Keep model-facing errors free of raw provider responses."""

    def decorate(
        method: Callable[Concatenate[TensorlakeFilesystemBackend, _Params], _Result],
    ) -> Callable[Concatenate[TensorlakeFilesystemBackend, _Params], _Result]:
        @wraps(method)
        def call(
            backend: TensorlakeFilesystemBackend,
            /,
            *args: _Params.args,
            **kwargs: _Params.kwargs,
        ) -> _Result:
            try:
                return method(backend, *args, **kwargs)
            except ValueError as error:
                return result_type(error=str(error))
            except FilesystemError as error:
                return result_type(error=_storage_error(error))

        return call

    return decorate


class TensorlakeFilesystemBackend(BackendProtocol):
    """Shared file tools without a shell or a sandbox dependency.

    Pass a Tensorlake ``Filesystem`` or a synchronous filesystem resolver.
    All paths are rooted at that filesystem. CompositeBackend may mount this
    backend under any virtual prefix. File tools never select a different volume.

    Args:
        filesystem: Existing filesystem or resolver called within each operation.
        max_file_bytes: Maximum bytes read or written by a single file operation.
        max_search_files: Maximum entries visited by a directory search.
        max_search_bytes: Maximum content bytes inspected by grep.

    Operations are not serialized. Callers must coordinate conflicting writes:
    the existence check in ``write`` and read-modify-write in ``edit`` are not
    atomic. Cross-backend coordination, snapshot retention, sandbox mounts, and
    task lifecycle also belong to the caller.
    Async methods run synchronous SDK work in worker threads.
    """

    def __init__(
        self,
        filesystem: Filesystem | Callable[[], Filesystem],
        *,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_search_files: int = MAX_SEARCH_FILES,
        max_search_bytes: int = MAX_SEARCH_BYTES,
    ) -> None:
        """Wrap an existing Cloud Volume or a synchronous filesystem resolver."""
        if isinstance(filesystem, Filesystem):
            self._filesystem = lambda: filesystem
        elif callable(filesystem):
            self._filesystem = filesystem
        else:
            message = (
                "filesystem must be a Tensorlake Filesystem or synchronous resolver"
            )
            raise TypeError(message)
        for name, value in {
            "max_file_bytes": max_file_bytes,
            "max_search_files": max_search_files,
            "max_search_bytes": max_search_bytes,
        }.items():
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        self._max_file_bytes = max_file_bytes
        self._max_search_files = max_search_files
        self._max_search_bytes = max_search_bytes

    def _path(self, path: str, *, allow_root: bool = False) -> str:
        return _volume_path(path, allow_root=allow_root)

    @_file_result(LsResult)
    def ls(self, path: str) -> LsResult:
        """List immediate children at one volume version, excluding symlinks."""
        relative = self._path(path, allow_root=True)
        fs = self._filesystem()
        version = fs.status().version_id
        if version is None:
            return LsResult(entries=[])
        entries: list[FileInfo] = []
        for row in fs.list_files(relative, version=version):
            if row.is_symlink:
                continue
            self._path("/" + row.path)
            entries.append(
                {
                    "path": "/" + row.path.rstrip("/") + ("/" if row.is_dir else ""),
                    "is_dir": row.is_dir,
                    "size": row.size or 0,
                }
            )
        return LsResult(entries=entries)

    def _read_bytes(
        self, fs: Filesystem, path: str, version: str | None = None
    ) -> bytes:
        if version is None:
            version = fs.status().version_id
        if version is None:
            raise FileNotFoundInFilesystemError(fs.name, path)
        data = fs.read_file(path, version=version)
        if len(data) > self._max_file_bytes:
            message = (
                f"file_too_large: maximum file size is {self._max_file_bytes} bytes"
            )
            raise ValueError(message)
        return data

    @_file_result(ReadResult)
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read paginated UTF-8 text or a supported base64 binary preview."""
        relative = self._path(file_path)
        if int(limit) <= 0:
            return empty_read_result()
        data = self._read_bytes(self._filesystem(), relative)
        if _get_backend_read_file_type(file_path) != "text":
            if len(data) > 500 * 1024:
                return ReadResult(error="Binary file exceeds maximum preview size")
            return ReadResult(
                file_data=FileData(
                    content=base64.b64encode(data).decode("ascii"), encoding="base64"
                )
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return ReadResult(error="binary_file: use download_files for raw bytes")
        return slice_read_response(
            FileData(content=text, encoding="utf-8"), offset, limit
        )

    @_file_result(WriteResult)
    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a UTF-8 file, returning an error if it already exists."""
        relative = self._path(file_path)
        data = content.encode("utf-8")
        if len(data) > self._max_file_bytes:
            return WriteResult(
                error=(
                    f"file_too_large: maximum file size is {self._max_file_bytes} bytes"
                )
            )
        fs = self._filesystem()
        version = fs.status().version_id
        if version is not None:
            try:
                fs.read_file(relative, version=version)
            except FileNotFoundInFilesystemError:
                pass
            else:
                return WriteResult(
                    error="file_exists: use edit_file for an existing file"
                )
        fs.write_file(relative, data)
        return WriteResult(path="/" + relative)

    @_file_result(EditResult)
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002 - BackendProtocol signature.
    ) -> EditResult:
        """Replace exact text matches, rejecting ambiguous edits by default."""
        relative = self._path(file_path)
        fs = self._filesystem()
        try:
            text = self._read_bytes(fs, relative).decode("utf-8")
        except UnicodeDecodeError:
            return EditResult(error="binary_file: cannot edit as text")
        result = perform_string_replacement(text, old_string, new_string, replace_all)
        if isinstance(result, str):
            return EditResult(error=result)
        changed, count = result
        if len(changed.encode("utf-8")) > self._max_file_bytes:
            return EditResult(
                error=(
                    f"file_too_large: maximum file size is {self._max_file_bytes} bytes"
                )
            )
        fs.write_file(relative, changed)
        return EditResult(path="/" + relative, occurrences=count)

    @_file_result(DeleteResult)
    def delete(self, file_path: str) -> DeleteResult:
        """Delete one volume path while rejecting the root directory."""
        relative = self._path(file_path)
        self._filesystem().delete_file(relative)
        return DeleteResult(path="/" + relative)

    def _walk(self, fs: Filesystem, path: str, version: str) -> Iterator[FileEntry]:
        pending = [path]
        visited = 0
        while pending:
            directory = pending.pop()
            for row in fs.list_files(directory, version=version):
                visited += 1
                if visited > self._max_search_files:
                    message = "search_too_large: narrow the search path"
                    raise ValueError(message)
                if row.is_symlink:
                    continue
                self._path("/" + row.path)
                if row.is_dir:
                    pending.append(row.path)
                else:
                    yield row

    def _grep_entries(
        self, fs: Filesystem, path: str, version: str
    ) -> Iterator[FileEntry]:
        if path:
            parent = str(PurePosixPath(path).parent)
            parent = "" if parent == "." else parent
            entry = next(
                (
                    row
                    for row in fs.list_files(parent, version=version)
                    if row.path == path
                ),
                None,
            )
            if entry is None or entry.is_symlink:
                raise FileNotFoundInFilesystemError(fs.name, path)
            if not entry.is_dir:
                yield entry
                return
        yield from self._walk(fs, path, version)

    @_file_result(GlobResult)
    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Match file paths using Deep Agents glob semantics."""
        relative = self._path(path or "/", allow_root=True)
        matcher = compile_grep_include_glob(pattern)
        fs = self._filesystem()
        version = fs.status().version_id
        if version is None:
            return GlobResult(matches=[])
        matches: list[FileInfo] = []
        for row in self._walk(fs, relative, version):
            name = str(PurePosixPath(row.path).relative_to(relative or "."))
            if matcher(name):
                matches.append(
                    {"path": "/" + row.path, "is_dir": False, "size": row.size or 0}
                )
        return GlobResult(matches=matches)

    @_file_result(GrepResult)
    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search for literal text within the configured file and byte budgets."""
        relative = self._path(path or "/", allow_root=True)
        matcher = compile_grep_include_glob(glob or "*")
        fs = self._filesystem()
        version = fs.status().version_id
        if version is None:
            return GrepResult(matches=[])
        matches: list[GrepMatch] = []
        total = 0
        maximum = max(0, max_count) if max_count is not None else 1000
        for row in self._grep_entries(fs, relative, version):
            if row.path == relative:
                name = PurePosixPath(row.path).name
            else:
                name = str(PurePosixPath(row.path).relative_to(relative or "."))
            if not matcher(name) or _get_backend_read_file_type(row.path) != "text":
                continue
            data = self._read_bytes(fs, row.path, version)
            total += len(data)
            if total > self._max_search_bytes:
                return GrepResult(matches=matches, truncated=True)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    if len(matches) >= maximum:
                        return GrepResult(matches=matches, truncated=True)
                    matches.append(
                        {"path": "/" + row.path, "line": number, "text": line}
                    )
        return GrepResult(matches=matches)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Validate a batch before publishing it with one SDK write."""
        try:
            uploads: dict[str, bytes | str] = {
                self._path(path): data for path, data in files
            }
            if any(len(data) > self._max_file_bytes for data in uploads.values()):
                return [
                    FileUploadResponse(path=path, error="invalid_path")
                    for path, _ in files
                ]
            if len(uploads) != len(files):
                return [
                    FileUploadResponse(path=path, error="invalid_path")
                    for path, _ in files
                ]
            if uploads:
                self._filesystem().write_files(uploads)
            return [FileUploadResponse(path=path) for path, _ in files]
        except ValueError:
            return [
                FileUploadResponse(path=path, error="invalid_path") for path, _ in files
            ]
        except FilesystemError as error:
            return [
                FileUploadResponse(path=path, error=_storage_error(error))
                for path, _ in files
            ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download raw bytes with an individual result for each requested path."""
        responses = []
        for path in paths:
            try:
                relative = self._path(path)
                data = self._read_bytes(self._filesystem(), relative)
                responses.append(FileDownloadResponse(path=path, content=data))
            except ValueError:
                responses.append(FileDownloadResponse(path=path, error="invalid_path"))
            except FilesystemError as error:
                responses.append(
                    FileDownloadResponse(path=path, error=_storage_error(error))
                )
        return responses
