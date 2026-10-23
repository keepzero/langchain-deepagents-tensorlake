"""Tensorlake sandbox backend implementation."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import (
    MAX_BINARY_BYTES,
    MAX_OUTPUT_BYTES,
    TRUNCATION_MSG,
    BaseSandbox,
)
from deepagents.backends.utils import _get_backend_read_file_type
from tensorlake.sandbox.exceptions import SandboxError as TensorlakeSandboxError

from langchain_deepagents_tensorlake._utils import slice_read_response

if TYPE_CHECKING:
    from tensorlake.sandbox import Sandbox as TensorlakeSandboxClient

logger = logging.getLogger(__name__)


class TensorlakeSandbox(BaseSandbox):
    """Tensorlake sandbox implementation conforming to SandboxBackendProtocol."""

    def __init__(
        self,
        sandbox: TensorlakeSandboxClient,
        *,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend wrapping an existing Tensorlake sandbox."""
        self._sandbox = sandbox
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return the sandbox id."""
        return self._sandbox.sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox."""
        effective_timeout = timeout if timeout is not None else self._default_timeout

        # Tensorlake sandbox expects `run()` to be called with an executable path and
        # optional args. The test command can be compound shell syntax (&&, |, etc.),
        # so run it through a shell interpreter.
        result = self._sandbox.run(
            "/bin/sh",
            args=["-c", command],
            timeout=effective_timeout,
        )

        output = result.stdout or ""
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write file contents through Tensorlake native write_file."""
        try:
            self._sandbox.read_file(file_path)
            return WriteResult(error=f"File '{file_path}' already exists")
        except TensorlakeSandboxError:
            pass
        try:
            self._sandbox.write_file(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path)
        except TensorlakeSandboxError as exc:
            return WriteResult(error=f"Failed to write file '{file_path}': {exc}")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read file content using the native read_file API."""
        try:
            raw_result = self._sandbox.read_file(file_path)
            raw: bytes = (
                raw_result.value if hasattr(raw_result, "value") else raw_result
            )
        except TensorlakeSandboxError as exc:
            return ReadResult(error=f"File '{file_path}': {exc}")

        is_binary = _get_backend_read_file_type(file_path) != "text"
        if not is_binary:
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                is_binary = True

        if is_binary:
            if len(raw) > MAX_BINARY_BYTES:
                return ReadResult(
                    error=(
                        f"File '{file_path}': Binary file exceeds"
                        f" maximum preview size of {MAX_BINARY_BYTES} bytes"
                    )
                )
            return ReadResult(
                file_data=FileData(
                    content=base64.b64encode(raw).decode("ascii"),
                    encoding="base64",
                )
            )

        result = slice_read_response(
            FileData(content=raw.decode("utf-8"), encoding="utf-8"), offset, limit
        )
        if result.file_data is None:
            return result
        page = result.file_data["content"]
        encoded = page.encode("utf-8")
        msg_bytes = len(TRUNCATION_MSG.encode("utf-8"))
        if len(encoded) > MAX_OUTPUT_BYTES - msg_bytes:
            prefix = encoded[: MAX_OUTPUT_BYTES - msg_bytes].decode(
                "utf-8", errors="ignore"
            )
            # Resume at the partial boundary line. Advance for an oversized first
            # line, matching BaseSandbox's pagination progress guarantee.
            next_offset = max(0, int(offset)) + max(1, prefix.count("\n"))
            return ReadResult(
                file_data=FileData(content=prefix + TRUNCATION_MSG, encoding="utf-8"),
                total_lines=result.total_lines,
                start_line=result.start_line,
                end_line=next_offset,
                next_offset=(
                    next_offset
                    if result.total_lines is not None
                    and next_offset < result.total_lines
                    else None
                ),
            )
        return result

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Read through the native SDK with the same pagination as sync reads."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Use the same native create-only operation for async writes."""
        return await asyncio.to_thread(self.write, file_path, content)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox using the native read_file API."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            try:
                raw_result = self._sandbox.read_file(path)
                content: bytes = (
                    raw_result.value if hasattr(raw_result, "value") else raw_result
                )
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except TensorlakeSandboxError as exc:
                error = (
                    "permission_denied"
                    if "permission" in str(exc).lower()
                    else "file_not_found"
                )
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=error)
                )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the Tensorlake sandbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._sandbox.write_file(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except TensorlakeSandboxError as exc:
                logger.warning("Tensorlake upload failed for %s: %s", path, exc)
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
        return responses
