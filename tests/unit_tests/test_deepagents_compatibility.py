"""Exercise native sandbox reads and the installed Deep Agents tool pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pytest
from deepagents import create_deep_agent
from deepagents.backends.sandbox import MAX_OUTPUT_BYTES, TRUNCATION_MSG
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from tensorlake.filesystem import FileEntry, Filesystem
from tensorlake.filesystem.exceptions import FileNotFoundInFilesystemError
from tensorlake.sandbox.exceptions import SandboxError

from langchain_deepagents_tensorlake import (
    TensorlakeFilesystemBackend,
    TensorlakeSandbox,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable


class _ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self.bind(tools=tools, **kwargs)


@pytest.fixture
def sandbox_backend() -> tuple[TensorlakeSandbox, MagicMock]:
    client = MagicMock()
    client.sandbox_id = "test-sandbox"
    return TensorlakeSandbox(client), client


def test_sandbox_read_preserves_newlines_and_pagination(
    sandbox_backend: tuple[TensorlakeSandbox, MagicMock],
) -> None:
    backend, client = sandbox_backend
    client.read_file.return_value = b"one\r\ntwo\r\nthree\n"
    result = backend.read("/report.md", offset=1, limit=1)
    assert result.file_data == {"content": "two\n", "encoding": "utf-8"}
    assert result.total_lines == 3  # noqa: PLR2004
    assert result.start_line == result.end_line == result.next_offset == 2  # noqa: PLR2004
    assert (
        backend.read("/report.md", offset=-1, limit=1).file_data["content"] == "one\n"
    )
    empty = backend.read("/report.md", limit=0)
    assert empty.error is None
    assert empty.file_data == {"content": "", "encoding": "utf-8"}
    assert backend.read("/report.md", offset=100).error
    client.run.assert_not_called()


def test_sandbox_read_truncation_resumes_at_partial_line(
    sandbox_backend: tuple[TensorlakeSandbox, MagicMock],
) -> None:
    backend, client = sandbox_backend
    # Each line fits on its own, but the requested page exceeds the byte cap.
    line = "你好" * (MAX_OUTPUT_BYTES // 20) + "\n"
    client.read_file.return_value = (line * 4).encode()
    result = backend.read("/large.md")
    assert len(result.file_data["content"].encode()) <= MAX_OUTPUT_BYTES
    assert result.file_data["content"].endswith(TRUNCATION_MSG)
    assert result.next_offset == 3  # noqa: PLR2004
    remainder = backend.read("/large.md", offset=result.next_offset)
    assert remainder.file_data["content"] == line
    assert remainder.next_offset is None


def test_sandbox_supported_media_remains_binary(
    sandbox_backend: tuple[TensorlakeSandbox, MagicMock],
) -> None:
    backend, client = sandbox_backend
    client.read_file.return_value = b"ascii bytes in a video file"
    assert backend.read("/clip.mp4").file_data["encoding"] == "base64"


async def test_sandbox_async_read_and_write_use_native_sdk(
    sandbox_backend: tuple[TensorlakeSandbox, MagicMock],
) -> None:
    backend, client = sandbox_backend
    client.read_file.return_value = b"one\ntwo\n"
    result = await backend.aread("/report.md", offset=1, limit=1)
    assert result.file_data["content"] == "two\n"
    assert result.total_lines == 2  # noqa: PLR2004
    assert (await backend.awrite("/report.md", "new")).error
    client.write_file.assert_not_called()
    client.read_file.side_effect = SandboxError("file not found")
    assert (await backend.awrite("/new.md", "new")).path == "/new.md"
    client.write_file.assert_called_once_with("/new.md", b"new")
    client.run.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit", [0, 1])
async def test_filesystem_backend_with_deep_agent_tools(
    *, asynchronous: bool, limit: int
) -> None:
    filesystem = Mock(spec=Filesystem)
    filesystem.name = "test-volume"
    filesystem.status.return_value = Mock(version_id="head")
    filesystem.read_file.side_effect = [
        FileNotFoundInFilesystemError("test-volume", "report.md"),
        b"first\nsecond\n",
    ]
    backend = TensorlakeFilesystemBackend(filesystem)
    model = _ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {
                            "file_path": "/report.md",
                            "content": "first\nsecond\n",
                        },
                        "id": "write-report",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {
                            "file_path": "/report.md",
                            "offset": 1,
                            "limit": limit,
                        },
                        "id": "read-report",
                    }
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    agent = create_deep_agent(model=model, backend=backend)
    inputs = {"messages": [{"role": "user", "content": "Write and read a report."}]}
    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2  # noqa: PLR2004
    assert all(m.status == "success" for m in tool_messages)
    if limit:
        assert "second" in str(tool_messages[-1].content)
    else:
        assert filesystem.read_file.call_count == 1
    filesystem.write_file.assert_called_once_with("report.md", b"first\nsecond\n")
    filesystem.read_file.assert_called_with("report.md", version="head")


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("*.{md,txt}", {"/a.md", "/nested/b.txt"}),
        ("/*.md", {"/a.md"}),
        ("**/*.txt", {"/nested/b.txt"}),
        ("/nested/*.txt", {"/nested/b.txt"}),
    ],
)
def test_filesystem_glob_semantics_across_versions(
    pattern: str, expected: set[str]
) -> None:
    filesystem = Mock(spec=Filesystem)
    filesystem.status.return_value = Mock(version_id="head")
    filesystem.list_files.side_effect = lambda path, version: {  # noqa: ARG005
        "": [
            FileEntry(name="a.md", path="a.md"),
            FileEntry(name="nested", path="nested", kind="directory"),
        ],
        "nested": [FileEntry(name="b.txt", path="nested/b.txt")],
    }[path]
    backend = TensorlakeFilesystemBackend(filesystem)
    result = backend.glob(pattern)
    assert result.error is None
    assert {entry["path"] for entry in result.matches} == expected


@pytest.mark.parametrize("pattern", ["../*.md", "{1..1001}"])
def test_invalid_glob_is_rejected_before_storage_access(pattern: str) -> None:
    filesystem = Mock(spec=Filesystem)
    backend = TensorlakeFilesystemBackend(filesystem)
    assert backend.glob(pattern).error
    assert backend.grep("needle", glob=pattern).error
    filesystem.status.assert_not_called()
