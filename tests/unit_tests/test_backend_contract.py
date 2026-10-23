"""Portable backend policies and the asynchronous BackendProtocol surface."""

import asyncio
from contextvars import ContextVar
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from tensorlake.filesystem import FileEntry, Filesystem
from tensorlake.filesystem.exceptions import FilesystemAPIError

from langchain_deepagents_tensorlake import TensorlakeFilesystemBackend
from langchain_deepagents_tensorlake.backends import (
    TensorlakeFilesystemBackend as Exported,
)


@pytest.fixture
def filesystem() -> Mock:
    filesystem = Mock(spec=Filesystem)
    filesystem.name = "test-volume"
    filesystem.status.return_value = SimpleNamespace(version_id="head")
    return filesystem


def test_concrete_filesystem_and_public_exports(filesystem: Mock) -> None:
    backend = TensorlakeFilesystemBackend(filesystem)
    filesystem.read_file.return_value = b"content"
    assert isinstance(backend, BackendProtocol)
    assert Exported is TensorlakeFilesystemBackend
    assert backend.read("/report.md").file_data["content"] == "content"
    assert not hasattr(backend, "execute")
    filesystem.snapshot.assert_not_called()


def test_dot_directories_are_visible(filesystem: Mock) -> None:
    filesystem.read_file.return_value = b"visible"
    filesystem.list_files.return_value = [
        FileEntry(name=".mytrade", path=".mytrade", kind="directory")
    ]
    backend = TensorlakeFilesystemBackend(filesystem)
    assert backend.read("/.mytrade/report.md").error is None
    assert backend.ls("/").entries[0]["path"] == "/.mytrade/"


def test_nested_dot_directories_are_searchable_and_symlinks_are_skipped(
    filesystem: Mock,
) -> None:
    backend = TensorlakeFilesystemBackend(filesystem)
    filesystem.read_file.return_value = b"visible"
    filesystem.list_files.side_effect = lambda path, version: {
        "": [FileEntry(name="reports", path="reports", kind="directory")],
        "reports": [
            FileEntry(name=".internal", path="reports/.internal", kind="directory"),
            FileEntry(name="visible.txt", path="reports/visible.txt"),
            FileEntry(name="link", path="reports/link", kind="symlink"),
        ],
        "reports/.internal": [FileEntry(name="a.txt", path="reports/.internal/a.txt")],
    }[path]
    assert backend.read("/reports/.internal/a.txt").file_data["content"] == "visible"
    assert {row["path"] for row in backend.ls("/reports").entries} == {
        "/reports/.internal/",
        "/reports/visible.txt",
    }
    expected = {"/reports/visible.txt", "/reports/.internal/a.txt"}
    assert {
        row["path"] for row in backend.glob("*.txt", path="/reports").matches
    } == expected
    assert {
        row["path"] for row in backend.grep("visible", path="/reports").matches
    } == expected
    assert [
        row["path"] for row in backend.glob(".internal/*.txt", path="/reports").matches
    ] == ["/reports/.internal/a.txt"]


@pytest.mark.parametrize(
    "option", ["max_file_bytes", "max_search_files", "max_search_bytes"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_processing_budgets_require_positive_integers(
    filesystem: Mock, option: str, value: object
) -> None:
    with pytest.raises(ValueError, match=option):
        TensorlakeFilesystemBackend(filesystem, **{option: value})


def test_configurable_file_limit_and_empty_upload(filesystem: Mock) -> None:
    backend = TensorlakeFilesystemBackend(filesystem, max_file_bytes=3)
    filesystem.status.return_value = SimpleNamespace(version_id=None)
    assert "file_too_large" in backend.write("/a", "four").error
    assert backend.upload_files([]) == []
    assert backend.upload_files([("/a", b"four")])[0].error
    filesystem.write_files.assert_not_called()
    filesystem.status.return_value = SimpleNamespace(version_id="head")
    filesystem.read_file.return_value = b"four"
    assert "file_too_large" in backend.read("/a.txt").error


def test_atomic_upload_rejects_normalized_duplicate_paths(filesystem: Mock) -> None:
    backend = TensorlakeFilesystemBackend(filesystem)
    result = backend.upload_files([("/a/b", b"first"), ("/a//b", b"second")])
    assert all(row.error for row in result)
    filesystem.write_files.assert_not_called()


@pytest.mark.parametrize("path", ["/", "//", "//escape", "/~", "/a/../b"])
def test_invalid_file_paths_do_not_access_sdk(filesystem: Mock, path: str) -> None:
    backend = TensorlakeFilesystemBackend(filesystem)
    assert backend.read(path).error
    assert backend.write(path, "x").error
    assert backend.edit(path, "a", "b").error
    assert backend.delete(path).error
    filesystem.status.assert_not_called()
    filesystem.delete_file.assert_not_called()


def test_search_entry_and_byte_budgets(filesystem: Mock) -> None:
    filesystem.list_files.return_value = [
        FileEntry(name="a.txt", path="a.txt"),
        FileEntry(name="b.txt", path="b.txt"),
    ]
    filesystem.read_file.return_value = b"match"
    backend = TensorlakeFilesystemBackend(filesystem, max_search_files=1)
    assert "search_too_large" in backend.glob("*").error
    backend = TensorlakeFilesystemBackend(filesystem, max_search_bytes=5)
    result = backend.grep("match")
    assert result.truncated
    assert len(result.matches) == 1
    assert all(
        row.kwargs["version"] == "head" for row in filesystem.read_file.call_args_list
    )


def test_grep_is_literal_and_file_filters_are_respected(filesystem: Mock) -> None:
    filesystem.list_files.return_value = [
        FileEntry(name="a.md", path="a.md"),
        FileEntry(name="b.txt", path="b.txt"),
    ]
    filesystem.read_file.return_value = b"[literal]\nordinary\n"
    backend = TensorlakeFilesystemBackend(filesystem)
    assert backend.grep("[literal]", glob="*.md").matches == [
        {"path": "/a.md", "line": 1, "text": "[literal]"}
    ]
    filesystem.read_file.assert_called_once_with("a.md", version="head")


def test_binary_download_partial_failure_and_redaction(filesystem: Mock) -> None:
    backend = TensorlakeFilesystemBackend(filesystem)
    filesystem.read_file.side_effect = [
        b"\x00\xff",
        FilesystemAPIError(403, "secret-provider-token"),
    ]
    result = backend.download_files(["/a.bin", "/bad.bin", "/../escape"])
    assert result[0].content == b"\x00\xff"
    assert result[1].error == "permission_denied"
    assert result[2].error == "invalid_path"
    assert "secret-provider-token" not in repr(result)


def test_create_only_write_edit_and_delete(filesystem: Mock) -> None:
    backend = TensorlakeFilesystemBackend(filesystem)
    filesystem.read_file.return_value = b"old"
    assert "file_exists" in backend.write("/file.txt", "new").error
    assert backend.edit("/file.txt", "old", "new").occurrences == 1
    filesystem.write_file.assert_called_once_with("file.txt", "new")
    assert backend.delete("/file.txt").path == "/file.txt"
    filesystem.delete_file.assert_called_once_with("file.txt")
    filesystem.snapshot.assert_not_called()


async def test_independent_async_file_operations_can_overlap(filesystem: Mock) -> None:
    ready = Barrier(3, timeout=5)

    def resolve() -> Filesystem:
        ready.wait()
        return filesystem

    filesystem.read_file.return_value = b"content"
    backend = TensorlakeFilesystemBackend(resolve)
    read, uploaded, downloaded = await asyncio.gather(
        backend.aread("/read.txt"),
        backend.aupload_files([("/upload.txt", b"new")]),
        backend.adownload_files(["/download.txt"]),
    )

    assert read.error is None
    assert read.file_data["content"] == "content"
    assert uploaded[0].error is None
    filesystem.write_files.assert_called_once_with({"upload.txt": b"new"})
    assert downloaded[0].error is None
    assert downloaded[0].content == b"content"


async def test_async_methods_preserve_resolver_context_and_composite_route(
    filesystem: Mock,
) -> None:
    current = ContextVar("current_filesystem")
    current.set(filesystem)
    filesystem.read_file.return_value = b"one\ntwo\n"
    filesystem.list_files.return_value = [FileEntry(name="a.md", path="a.md")]
    backend = TensorlakeFilesystemBackend(current.get)
    composite = CompositeBackend(
        default=StateBackend(), routes={"/artifacts/": backend}
    )
    assert (await composite.aread("/artifacts/a.md", offset=1, limit=1)).file_data[
        "content"
    ] == "two\n"
    assert (await backend.als("/")).entries[0]["path"] == "/a.md"
    assert (await backend.aglob("*.md")).matches[0]["path"] == "/a.md"
    assert (await backend.agrep("two")).matches[0]["line"] == 2
    assert (await backend.agrep("two", max_count=0)).truncated
    assert (await composite.agrep("two", path="/artifacts/")).matches[0][
        "path"
    ] == "/artifacts/a.md"
    assert (await backend.aedit("/a.md", "two", "changed")).occurrences == 1
    assert (await backend.adelete("/a.md")).error is None
    assert (await backend.aupload_files([("/b", b"binary")]))[0].error is None
    assert (await backend.adownload_files(["/a.md"]))[0].content == b"one\ntwo\n"
    filesystem.status.return_value = SimpleNamespace(version_id=None)
    assert (await backend.awrite("/new.md", "first")).error is None
