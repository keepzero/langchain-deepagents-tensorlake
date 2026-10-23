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

Use absolute POSIX paths within the supplied volume. Traversal paths are
rejected; dotfiles are supported, and symlinks are skipped in listings and
searches. Root is only valid for directory operations.

- Text reads support pagination; binary previews are limited to 500 KiB.
  Use downloads for raw bytes. Reads and searches use a fixed volume version.
- `write` creates files; `edit` requires a unique exact match unless
  `replace_all=True`. Uploads can replace existing files.
- Uploads validate the batch before one SDK write; downloads return a result
  for each path.

### Ownership, limits, and errors

The caller manages volumes, credentials, sandbox lifecycle, and retained
snapshots. Keep credentials outside agent-visible files.

The backend does not lock file operations. Callers must coordinate conflicting
access: `write` existence checks and `edit` read-modify-write are not atomic.
Async methods use worker threads; supplied filesystems and resolvers must
support concurrent calls or be serialized by the caller.

Constructor limits must be positive integers:

| Option | Default | Limit |
| --- | --- | --- |
| `max_file_bytes` | 32 MiB | Bytes read or written by one file operation |
| `max_search_files` | 10,000 | Entries visited during a search |
| `max_search_bytes` | 64 MiB | Content bytes inspected by grep |

These are processing limits, not storage quotas. Files are downloaded in full
before read-size checks. Storage errors omit raw provider details; authentication
and permission failures remain distinct from missing files.

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
