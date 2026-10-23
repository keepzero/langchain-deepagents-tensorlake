"""Real Cloud Volume tests; only the random volume created here is disposable."""

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from tensorlake.filesystem import Filesystem, FilesystemClient
from tensorlake.filesystem.exceptions import FilesystemError, FilesystemNotFoundError

from langchain_deepagents_tensorlake import TensorlakeFilesystemBackend

pytestmark = pytest.mark.integration


@pytest.fixture
def filesystem() -> Iterator[Filesystem]:
    api_key = os.environ.get("TENSORLAKE_API_KEY")
    if not api_key:
        pytest.fail(
            "Set TENSORLAKE_API_KEY to run real Cloud Volume tests", pytrace=False
        )
    client = FilesystemClient(
        api_key=api_key,
        api_url=os.environ.get("TENSORLAKE_API_URL", "https://api.tensorlake.ai"),
    )
    name = "langchain-tensorlake-test-" + uuid4().hex
    try:
        try:
            volume = client.create(name)
        except FilesystemError as error:
            pytest.fail(
                f"Could not create test Cloud Volume ({type(error).__name__})",
                pytrace=False,
            )
        yield volume
    finally:
        try:
            client.delete(name)
        except FilesystemNotFoundError:
            pass
        except FilesystemError as error:
            pytest.fail(
                f"Could not clean test Cloud Volume ({type(error).__name__})",
                pytrace=False,
            )


def test_real_filesystem_contract_and_caller_owned_snapshot(
    filesystem: Filesystem,
) -> None:
    backend = TensorlakeFilesystemBackend(filesystem)
    assert backend.ls("/").entries == []
    assert backend.read("/missing.md").error == "file_not_found"
    assert backend.write("/input.md", "main-data\n").error is None
    assert "file_exists" in backend.write("/input.md", "replacement").error
    assert backend.edit("/input.md", "main", "research").occurrences == 1
    assert backend.read("/input.md").file_data["content"] == "research-data\n"
    uploads = backend.upload_files(
        [("/data/a.txt", b"[literal]\n"), ("/data/b.bin", b"\x00\xff")]
    )
    assert all(item.error is None for item in uploads)
    assert backend.download_files(["/data/b.bin"])[0].content == b"\x00\xff"
    assert backend.glob("**/*.txt").matches[0]["path"] == "/data/a.txt"
    assert backend.grep("[literal]", path="/data/a.txt").matches[0]["line"] == 1
    assert {row["path"] for row in backend.ls("/data").entries} == {
        "/data/a.txt",
        "/data/b.bin",
    }
    retained = filesystem.snapshot("integration test caller retention")
    assert retained.id
    assert backend.delete("/input.md").error is None
    assert backend.read("/input.md").error == "file_not_found"
    assert filesystem.read_file("input.md", version=retained.id) == b"research-data\n"


async def test_real_async_file_operations(filesystem: Filesystem) -> None:
    backend = TensorlakeFilesystemBackend(lambda: filesystem)
    assert (await backend.awrite("/report.md", "first\nsecond\n")).error is None
    result = await backend.aread("/report.md", offset=1, limit=1)
    assert result.error is None
    assert result.file_data["content"] == "second\n"
    assert (await backend.aedit("/report.md", "second", "updated")).error is None
    assert (await backend.agrep("updated")).matches[0]["line"] == 2
    assert (await backend.adelete("/report.md")).error is None
