# langchain-aspected

A [LangChain](https://python.langchain.com/) `VectorStore` integration for [Aspected](https://aspected.com), a new kind of vector database that uses metadata as in-search
signals, not filters.

## Try Aspected locally

You can spin up a local instance of Aspected using Docker:

```bash
docker run -p 8080:8080 xillio/aspected:latest
```

This starts the Aspected server on `http://localhost:8080`, which you can point this client at. See the
[documentation](https://docs.aspected.com) for more information on getting started with the database setup.

## Installation

```bash
pip install langchain-aspected
```

Or, using `uv`:

```bash
uv add langchain-aspected
```

## Quick start

```python
from aspected_client import AspectedClient
from langchain_openai import OpenAIEmbeddings
from langchain_aspected import AspectedVectorStore

# Connect to a running Aspected server
client = AspectedClient(url="http://localhost:8080")
embeddings = OpenAIEmbeddings()

# Create an index, embed texts, and store them in one call
store = AspectedVectorStore.from_texts(
    texts=[
        "The quick brown fox jumps over the lazy dog",
        "LangChain makes building LLM apps easy",
        "Aspected is a multi-aspect vector database",
    ],
    embedding=embeddings,
    client=client,
    index_name="my-index",
)

# Semantic similarity search
results = store.similarity_search("vector database", k=2)
for doc in results:
    print(doc.page_content)

# Search with scores
results_with_scores = store.similarity_search_with_score("language model", k=2)
for doc, score in results_with_scores:
    print(f"[{score:.4f}] {doc.page_content}")
```

## Usage

### Connecting to an existing index

```python
from aspected_client import AspectedClient
from langchain_openai import OpenAIEmbeddings
from langchain_aspected import AspectedVectorStore

store = AspectedVectorStore(
    client=AspectedClient(url="http://localhost:8080"),
    embedding=OpenAIEmbeddings(),
    index_name="my-existing-index",
)
```

When the index already exists, its distance type, embedding aspect, and
schema are validated against this configuration on the next write; a
mismatch raises a `ValueError` instead of silently reusing the index.

### Building a store from an existing index automatically

Instead of manually re-specifying `distance_type` and `schema`, you can
derive them straight from the server. You can pass a pre-built `client`, or
just a `url` and let it construct one for you (like `from_texts`/`from_documents`):

```python
store = AspectedVectorStore.from_existing_index(
    embedding=OpenAIEmbeddings(),
    index_name="my-existing-index",
    url="http://localhost:8080",
)
```

### Adding documents

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="Hello world", metadata={"source": "example.txt"}),
    Document(page_content="Foo bar", metadata={"source": "other.txt"}),
]

ids = store.add_documents(docs)
```

### Deleting documents

```python
store.delete(ids=["id-1", "id-2"])
```

### Dropping the index

```python
store.delete_index()
```

## Configuration

| Parameter               | Default         | Description                                                             |
|-------------------------|-----------------|-------------------------------------------------------------------------|
| `client`                | required        | `AspectedClient` instance                                               |
| `embedding`             | required        | LangChain `Embeddings` instance                                         |
| `index_name`            | `"langchain"`   | Aspected index name                                                     |
| `content_payload_key`   | `"__payload"`   | Doc field key used to store raw text                                    |
| `content_embedding_key` | `"__embedding"` | Aspected aspect name for the embedding vector                           |
| `distance_type`         | `Cosine`        | Distance metric (`Cosine`, `Euclidean`, `DotProduct`, etc.)             |
| `id_size`               | `36`            | Size (in bytes) of document IDs used when creating a new index          |
| `schema`                | `None`          | Optional list of `AspectSchema` for additional required metadata fields |

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

```bash
uv sync --all-extras
```

### Running the unit tests

The unit test suite mocks the underlying `AspectedClient`, so no server is required:

```bash
uv run pytest
```

### Running the integration tests

Integration tests exercise every `AspectedVectorStore` operation against a
*real* Aspected server, started automatically in Docker via
[testcontainers](https://testcontainers-python.readthedocs.io/). They require
a working Docker daemon on the machine running them.

```bash
# Requires Docker
uv run pytest -m integration
```

The server image (and tag) used is configurable through the `ASPECTED_IMAGE`
environment variable, and defaults to `xillio/aspected:latest`:

```bash
ASPECTED_IMAGE=xillio/aspected:latest uv run pytest -m integration
```

`ASPECTED_STARTUP_TIMEOUT` (seconds, default `120`) controls how long to wait
for the server container to become ready.

### Checks

The same checks run in CI (see `.github/workflows/ci.yml`):

```bash
uv run ruff format --check              # formatting
uv run ruff check                       # lint
uv run ty check                         # type check
uv run pytest                           # unit tests
uv run pytest -m integration            # integration tests (requires Docker)
```

Configuration for `ty` lives under `[tool.ty]` in `pyproject.toml`.

## License

MIT
