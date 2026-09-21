"""Integration tests for AspectedVectorStore against a real Aspected server.

These tests spin up an actual Aspected server (via `testcontainers`, see
``conftest.py``) and exercise every public operation of
:class:`~langchain_aspected.AspectedVectorStore` end-to-end: index creation,
upserting, similarity search (with/without scores, by text/by vector),
deletion, schema validation, existing-index introspection, and index
deletion.

Run with:

    pytest -m integration

These tests require Docker to be available on the host running them.
"""

from __future__ import annotations

import hashlib

import pytest
from aspected_client import AspectedClient, DistanceType
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_aspected import AspectedVectorStore, AspectSchema

pytestmark = pytest.mark.integration


def _stable_hash(word: str) -> int:
    """A hash function that is stable across processes (unlike ``hash()``)."""
    return int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)


class DeterministicEmbeddings(Embeddings):
    """A tiny bag-of-words embedding model with no external dependencies.

    Produces normalized fixed-length vectors where texts sharing more words
    end up closer together, which is enough to meaningfully exercise
    similarity search against a real vector database.
    """

    def __init__(self, dims: int = 16) -> None:
        self._dims = dims

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dims
        for word in text.lower().split():
            vector[_stable_hash(word) % self._dims] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def embeddings() -> DeterministicEmbeddings:
    return DeterministicEmbeddings()


TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "LangChain makes building LLM applications easy",
    "Aspected is a multi aspect vector database",
]
METADATAS = [
    {"source": "animals.txt", "page": 1},
    {"source": "langchain.txt", "page": 2},
    {"source": "aspected.txt", "page": 3},
]


# ---------------------------------------------------------------------------
# add_texts / similarity_search
# ---------------------------------------------------------------------------


def test_add_texts_and_similarity_search(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    ids = store.add_texts(TEXTS, metadatas=METADATAS)
    assert len(ids) == len(TEXTS)

    results = store.similarity_search("vector database aspected", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].page_content == TEXTS[2]
    assert results[0].metadata["source"] == "aspected.txt"


def test_similarity_search_with_score_orders_by_distance(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    store.add_texts(TEXTS, metadatas=METADATAS)

    results = store.similarity_search_with_score("LangChain building applications", k=3)
    assert len(results) == 3
    distances = [score for _, score in results]
    assert distances == sorted(distances)
    assert results[0][0].page_content == TEXTS[1]


def test_similarity_search_by_vector(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    store.add_texts(TEXTS, metadatas=METADATAS)

    query_vector = embeddings.embed_query("lazy dog fox")
    docs = store.similarity_search_by_vector(query_vector, k=1)
    assert len(docs) == 1
    assert docs[0].page_content == TEXTS[0]


def test_add_texts_with_custom_ids(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    custom_ids = ["doc-a", "doc-b"]
    returned = store.add_texts(TEXTS[:2], ids=custom_ids)
    assert returned == custom_ids


def test_add_texts_upsert_replaces_existing_document(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    store.add_texts(["original text"], ids=["dup-id"])
    store.add_texts(["updated text"], ids=["dup-id"])

    docs = store.similarity_search("updated text", k=1)
    assert docs[0].page_content == "updated text"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_removes_documents(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    ids = store.add_texts(TEXTS, metadatas=METADATAS)

    result = store.delete([ids[0]])
    assert result is True

    remaining = store.similarity_search("fox dog", k=len(TEXTS))
    assert all(doc.page_content != TEXTS[0] for doc in remaining)


# ---------------------------------------------------------------------------
# from_texts / from_documents
# ---------------------------------------------------------------------------


def test_from_texts_creates_index_and_populates_it(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore.from_texts(
        texts=TEXTS,
        embedding=embeddings,
        metadatas=METADATAS,
        client=aspected_client,
        index_name=index_name,
    )
    docs = store.similarity_search("fox dog", k=1)
    assert docs[0].page_content == TEXTS[0]


def test_from_documents_creates_index_and_populates_it(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(TEXTS, METADATAS, strict=True)
    ]
    store = AspectedVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        client=aspected_client,
        index_name=index_name,
    )
    docs = store.similarity_search("LangChain applications", k=1)
    assert docs[0].page_content == TEXTS[1]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_add_texts_with_schema_round_trips_metadata(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
        schema=[AspectSchema(name="page", type="number", settings={"max": 1000})],
    )
    store.add_texts(["hello world"], metadatas=[{"page": 7}])

    docs = store.similarity_search("hello world", k=1)
    assert docs[0].metadata["page"] == 7


def test_add_texts_with_schema_missing_field_raises(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
        schema=[AspectSchema(name="page", type="number", settings={"max": 1000})],
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello world"], metadatas=[{}])


# ---------------------------------------------------------------------------
# existing-index validation
# ---------------------------------------------------------------------------


def test_reusing_existing_index_with_matching_config_succeeds(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store_a = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    store_a.add_texts(["first document"])

    # A second store pointed at the same, already-created index.
    store_b = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    store_b.add_texts(["second document"])

    docs = store_b.similarity_search("first document", k=2)
    assert {d.page_content for d in docs} == {"first document", "second document"}


def test_reusing_existing_index_with_mismatched_distance_type_raises(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store_a = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
        distance_type=DistanceType.cosine,
    )
    store_a.add_texts(["seed document"])

    store_b = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
        distance_type=DistanceType.euclidean,
    )
    with pytest.raises(ValueError):
        store_b.add_texts(["another document"])


# ---------------------------------------------------------------------------
# from_existing_index
# ---------------------------------------------------------------------------


def test_from_existing_index_reads_config_from_server(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    seed_store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
        schema=[AspectSchema(name="page", type="number", settings={"max": 1000})],
    )
    seed_store.add_texts(["seed document"], metadatas=[{"page": 3}])

    store = AspectedVectorStore.from_existing_index(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    assert store._distance_type == DistanceType.cosine
    assert any(aspect.name == "page" for aspect in store._schema)

    docs = store.similarity_search("seed document", k=1)
    assert docs[0].page_content == "seed document"


def test_from_existing_index_missing_index_raises(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    with pytest.raises(Exception):  # noqa: B017 - server raises its own not-found error
        AspectedVectorStore.from_existing_index(
            client=aspected_client,
            embedding=embeddings,
            index_name=index_name,
        )


# ---------------------------------------------------------------------------
# delete_index
# ---------------------------------------------------------------------------


def test_delete_index_removes_index(
    aspected_client: AspectedClient, embeddings: DeterministicEmbeddings, index_name: str
) -> None:
    store = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    store.add_texts(["will be deleted"])

    assert store.delete_index() is True

    # The index no longer exists, so a fresh store must recreate it.
    recreated = AspectedVectorStore(
        client=aspected_client,
        embedding=embeddings,
        index_name=index_name,
    )
    recreated.add_texts(["fresh start"])
    docs = recreated.similarity_search("fresh start", k=1)
    assert docs[0].page_content == "fresh start"
