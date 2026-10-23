# langchain-deepagents-tensorlake

[Tensorlake](https://tensorlake.ai) sandbox provider and Cloud Volume filesystem backends
for [Deep Agents](https://github.com/langchain-ai/deepagents).

## Installation

```bash
pip install langchain-deepagents-tensorlake
```

To use this checkout, including the merged filesystem backend:

```bash
uv sync --locked --all-groups
```

Requires Python 3.11+ and Deep Agents `>=0.7`. Both backends are included
in this distribution; a separate `langchain-tensorlake-backend` installation is
not needed. Existing filesystem users should change their import to
`from langchain_deepagents_tensorlake import TensorlakeFilesystemBackend`.

## Sandbox

Use `TensorlakeSandbox` for shell execution and file tools inside a Tensorlake
sandbox:

```python
from tensorlake.sandbox import Sandbox
from langchain_deepagents_tensorlake import TensorlakeSandbox
from deepagents.backends import CompositeBackend

# Create a Tensorlake sandbox
sandbox = Sandbox.create(api_key="your-api-key")
backend = TensorlakeSandbox(sandbox=sandbox)

# Use directly as a DeepAgents backend
# or wrap with CompositeBackend for routing
```

With the Deep Agents CLI (`deepagents-code`), set:

```bash
export TENSORLAKE_API_KEY=your-api-key
deepagents-code --sandbox tensorlake
```

> **Note:** CLI integration requires `deepagents-code` to be configured with
> a custom `sandbox_factory` that maps `"tensorlake"` to `TensorlakeSandbox`.
> See the [examples](examples/) directory for a complete setup.

The caller manages sandbox creation and termination. Mounting a Cloud Volume
inside a sandbox and coordinating access with a filesystem backend are also
application responsibilities.

## Cloud Volume filesystem

Use `TensorlakeFilesystemBackend` for file tools backed by an existing Cloud
Volume, including when no sandbox is running. Set `TENSORLAKE_API_KEY` in the
trusted application environment, then select the volume:

```python
from deepagents import create_deep_agent
from tensorlake.filesystem import FilesystemClient

from langchain_deepagents_tensorlake import TensorlakeFilesystemBackend

filesystem = FilesystemClient().get("agent-workspace")
backend = TensorlakeFilesystemBackend(filesystem)
agent = create_deep_agent(model="openai:gpt-4.1", backend=backend)

# You can also use the backend directly.
result = backend.read("/report.md")
if result.error:
    raise RuntimeError(result.error)
print(result.file_data["content"])
```

The backend supports `ls`, `read`, `write`, `edit`, `delete`, `glob`, `grep`, binary
upload/download, and their async counterparts. `write` creates a new file; use
`edit` or upload to replace an existing file. It accepts either a `Filesystem`
instance or a synchronous resolver. The application selects and manages volumes,
shared locks, and retained snapshots.

File operations return native Deep Agents result types, including pagination
metadata and search truncation. Deletion is available directly, through
`CompositeBackend`, and through the agent's built-in file tools.

It can also be mounted under a virtual prefix:

```python
from deepagents.backends import CompositeBackend, StateBackend

combined = CompositeBackend(default=StateBackend(), routes={"/workspace/": backend})
agent = create_deep_agent(model="openai:gpt-4.1", backend=combined)
```

### Paths and file operations

Paths are virtual absolute POSIX paths rooted within the supplied volume. Root
is accepted for directory operations; file operations reject root. Paths with
`..` or `~` components, a NUL, backslashes, or a double leading slash are rejected.
Repeated internal separators and `.` segments are normalized. Dotfiles and dot
directories are accessible; symlinks are omitted from listings and searches.
Basename glob patterns match at any depth, including inside dot directories;
recursive `**` patterns require explicitly matching dot-directory segments.

- Text reads preserve trailing newlines, normalize line endings to LF, and return
  pagination metadata. Negative offsets read from the beginning; non-positive
  limits perform no storage read.
  Supported binary previews return base64 with a 500 KiB limit; raw downloads
  preserve bytes. Invalid UTF-8 in text files returns a binary-file error.
- `write` is create-only and returns `file_exists` for an existing file. `edit`
  requires a unique exact match unless `replace_all=True`; upload can replace
  existing content. `delete` delegates one path to the SDK and rejects root.
- Uploads validate all paths and sizes, rejecting duplicate normalized paths,
  before one `write_files` publication. Downloads return one result per path.
- Reads and searches use explicit remote version IDs. Empty volumes are handled
  before attempting to access a nonexistent head.

### Ownership, limits, and errors

The backend does not serialize file operations or accept an external lock.
The existence check in `write` and the read-modify-write sequence in `edit`
are not atomic and do not use conditional writes. The application must
coordinate conflicting access to the same volume, including access through
other backend instances, sandbox commands, or workers.

Async methods run synchronous SDK work in worker threads. A filesystem resolver
is called inside each operation, with context variables propagated to those
threads. The supplied filesystem and resolver must support concurrent calls,
or the application must serialize access to them.

The caller selects and manages the volume, credentials, access policy, and
retained snapshots. The backend does not create volumes, start sandboxes, or
retain snapshots. File mutations produce working versions; the caller decides
when to retain a permanent snapshot. SDK writes, sandbox mounts, and application
database transactions are not one atomic operation. Keep API credentials outside
agent-visible files.

Processing budgets can be set in the constructor and must be positive integers:

| Option | Default | Limit |
| --- | --- | --- |
| `max_file_bytes` | 32 MiB | Bytes read or written by one file operation |
| `max_search_files` | 10,000 | Entries visited during a search |
| `max_search_bytes` | 64 MiB | Content bytes inspected by grep |

These are client processing budgets, not storage quotas. The SDK returns full
file bytes before the read-size check, so they do not bound network transfer or
allocation for an oversized remote file. Search errors or truncation indicate
that callers should narrow the search scope.

Storage errors are returned without raw provider messages, which may contain
credentials or signed URLs. Authentication and permission failures remain
distinct from missing files and are not treated as permission to create a file.

See the [Deep Agents backend documentation](https://docs.langchain.com/oss/python/deepagents/backends),
[Tensorlake Cloud Volume HTTP API](https://docs.tensorlake.ai/filesystems/http-api),
and [Tensorlake authentication](https://docs.tensorlake.ai/platform/authentication)
for the underlying APIs.

## Development

```bash
uv sync --locked --all-groups
make test
make lint
uv build
```

The default `uv run --group test pytest` runs unit tests only, without creating
remote resources. Real-service tests are selected explicitly and create and
clean up temporary volumes or sandboxes:

```bash
uv run --group test --env-file .env pytest tests/integration_tests/
```

## License

Apache 2.0
