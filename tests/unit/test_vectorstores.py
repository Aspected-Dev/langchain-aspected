"""Tests for the AspectedVectorStore LangChain integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from aspected_client import Aspect, DistanceType, Hnsw, Index
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_aspected import AspectedVectorStore, AspectSchema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings that return fixed-length float vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i) / 10 for i in range(4)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(i) / 10 for i in range(4)]


def make_client() -> MagicMock:
    """Return a mock AspectedClient."""
    client = MagicMock()
    # get_index raises so that _ensure_index creates the index
    client.get_index.side_effect = Exception("not found")
    # upload_docs returns a result with data
    upload_result = MagicMock()
    upload_result.data.created = 1
    upload_result.data.updated = 0
    client.upload_docs.return_value = upload_result
    # create_index returns a success result
    create_result = MagicMock()
    create_result.data.created = True
    client.create_index.return_value = create_result
    return client


def make_index(
    *,
    distance_type: DistanceType = DistanceType.cosine,
    aspects: list[Aspect] | None = None,
    num_dims: int = 4,
    id_size: int = 36,
) -> Index:
    """Build a real :class:`~aspected_client.Index` model for validation tests."""
    hnsw = Hnsw(
        M=16,
        Mmax=32,
        Mmax0=64,
        mL=0.25,
        efConstruction=200,
        efSearch=50,
        extendCandidates=False,
        keepPrunedConnections=False,
        maxStoreSize=1000,
    )
    return Index(
        name="test-index",
        idSize=id_size,
        numDims=num_dims,
        distanceType=distance_type,
        schema=aspects or [],
        hnsw=hnsw,
    )


# ---------------------------------------------------------------------------
# add_texts
# ---------------------------------------------------------------------------


def test_add_texts_returns_ids() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    ids = store.add_texts(["hello", "world"])
    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)


def test_add_texts_calls_upload_docs() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    store.add_texts(["hello"], metadatas=[{"source": "test"}])
    client.upload_docs.assert_called_once()


def test_add_texts_with_custom_ids() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    custom_ids = ["id-1", "id-2"]
    returned = store.add_texts(["a", "b"], ids=custom_ids)
    assert returned == custom_ids


def test_add_texts_empty() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    ids = store.add_texts([])
    assert ids == []
    client.upload_docs.assert_not_called()


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------


def _make_search_row(
    doc_id: str, text: str, distance: float, meta: dict | None = None
) -> MagicMock:
    row = MagicMock()
    row.id = doc_id
    row.distance = distance
    payload: dict = {"__payload": text, "__embedding": [0.1, 0.2], **(meta or {})}
    row.doc = payload
    return row


def test_similarity_search_returns_documents() -> None:
    client = make_client()
    search_result = MagicMock()
    search_result.data = [
        _make_search_row("id-1", "hello world", 0.05),
        _make_search_row("id-2", "foo bar", 0.42),
    ]
    client.search_index.return_value = search_result

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    docs = store.similarity_search("hello", k=2)
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "hello world"


def test_similarity_search_with_score() -> None:
    client = make_client()
    search_result = MagicMock()
    search_result.data = [_make_search_row("id-1", "hello", 0.1)]
    client.search_index.return_value = search_result

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    results = store.similarity_search_with_score("hello", k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert score == pytest.approx(0.1)


def test_similarity_search_by_vector() -> None:
    client = make_client()
    search_result = MagicMock()
    search_result.data = [_make_search_row("id-1", "content", 0.0)]
    client.search_index.return_value = search_result

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    docs = store.similarity_search_by_vector([0.1, 0.2, 0.3, 0.4], k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "content"


def test_metadata_preserved() -> None:
    client = make_client()
    search_result = MagicMock()
    row = _make_search_row("id-1", "content", 0.1, meta={"source": "wiki", "page": 3})
    search_result.data = [row]
    client.search_index.return_value = search_result

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    docs = store.similarity_search("query", k=1)
    assert docs[0].metadata.get("source") == "wiki"
    assert docs[0].metadata.get("page") == 3
    # embedding vector should NOT appear in metadata
    assert "__embedding" not in docs[0].metadata


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_calls_upload_docs() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    result = store.delete(["id-1", "id-2"])
    assert result is True
    client.upload_docs.assert_called_once()


def test_delete_empty_list() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    result = store.delete([])
    assert result is True
    client.upload_docs.assert_not_called()


# ---------------------------------------------------------------------------
# from_texts / from_documents
# ---------------------------------------------------------------------------


def test_from_texts_creates_store_and_adds_docs() -> None:
    client = make_client()
    store = AspectedVectorStore.from_texts(
        texts=["hello", "world"],
        embedding=FakeEmbeddings(),
        client=client,
        index_name="test-index",
    )
    assert isinstance(store, AspectedVectorStore)
    client.upload_docs.assert_called_once()


def test_from_documents() -> None:
    client = make_client()
    docs = [
        Document(page_content="first doc", metadata={"source": "a"}),
        Document(page_content="second doc", metadata={"source": "b"}),
    ]
    store = AspectedVectorStore.from_documents(
        documents=docs,
        embedding=FakeEmbeddings(),
        client=client,
        index_name="test-index",
    )
    assert isinstance(store, AspectedVectorStore)
    client.upload_docs.assert_called_once()


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_add_texts_with_schema_requires_all_fields() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        schema=[AspectSchema(name="source", type="keyword")],
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"], metadatas=[{}])


def test_add_texts_with_schema_succeeds_when_fields_present() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        schema=[AspectSchema(name="source", type="keyword")],
    )
    ids = store.add_texts(["hello"], metadatas=[{"source": "wiki"}])
    assert len(ids) == 1
    client.upload_docs.assert_called_once()


def test_ensure_index_includes_schema_aspects() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        schema=[AspectSchema(name="source", type="keyword", settings={"foo": "bar"})],
    )
    store.add_texts(["hello"], metadatas=[{"source": "wiki"}])
    client.create_index.assert_called_once()
    _, payload = client.create_index.call_args.args
    names = [aspect.name for aspect in payload.schema_]
    assert "__embedding" in names
    assert "source" in names
    source_aspect = next(a for a in payload.schema_ if a.name == "source")
    assert source_aspect.type == "keyword"
    assert source_aspect.settings == {"foo": "bar"}


def test_create_index_uses_default_id_size() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    store.add_texts(["hello"])
    _, payload = client.create_index.call_args.args
    assert payload.id_size == 36


def test_create_index_uses_custom_id_size() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        id_size=16,
    )
    store.add_texts(["hello"])
    _, payload = client.create_index.call_args.args
    assert payload.id_size == 16


def test_from_texts_with_schema() -> None:
    client = make_client()
    with pytest.raises(ValueError):
        AspectedVectorStore.from_texts(
            texts=["hello"],
            embedding=FakeEmbeddings(),
            client=client,
            index_name="test-index",
            metadatas=[{}],
            schema=[AspectSchema(name="source", type="keyword")],
        )


def test_schema_field_collides_with_content_payload_key() -> None:
    client = make_client()
    with pytest.raises(ValueError):
        AspectedVectorStore(
            client=client,
            embedding=FakeEmbeddings(),
            index_name="test-index",
            schema=[AspectSchema(name="__payload", type="keyword")],
        )


def test_schema_field_collides_with_content_embedding_key() -> None:
    client = make_client()
    with pytest.raises(ValueError):
        AspectedVectorStore(
            client=client,
            embedding=FakeEmbeddings(),
            index_name="test-index",
            schema=[AspectSchema(name="__embedding", type="keyword")],
        )


def test_content_payload_key_collides_with_content_embedding_key() -> None:
    client = make_client()
    with pytest.raises(ValueError):
        AspectedVectorStore(
            client=client,
            embedding=FakeEmbeddings(),
            index_name="test-index",
            content_payload_key="same",
            content_embedding_key="same",
        )


def test_custom_content_keys_do_not_collide_with_schema() -> None:
    client = make_client()
    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        schema=[AspectSchema(name="source", type="keyword")],
    )
    assert store is not None


# ---------------------------------------------------------------------------
# existing-index validation
# ---------------------------------------------------------------------------


def test_ensure_index_accepts_valid_existing_index() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    source_aspect = Aspect(name="source", type="keyword", path="$.source", settings={})
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(
        data=make_index(aspects=[embedding_aspect, source_aspect])
    )

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        schema=[AspectSchema(name="source", type="keyword")],
    )
    store.add_texts(["hello"], metadatas=[{"source": "wiki"}])
    client.create_index.assert_not_called()


def test_ensure_index_rejects_distance_type_mismatch() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(
        data=make_index(distance_type=DistanceType.euclidean, aspects=[embedding_aspect])
    )

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"])


def test_ensure_index_rejects_id_size_mismatch() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(
        data=make_index(aspects=[embedding_aspect], id_size=16)
    )

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"])


def test_ensure_index_accepts_matching_custom_id_size() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(
        data=make_index(aspects=[embedding_aspect], id_size=16)
    )

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        id_size=16,
    )
    store.add_texts(["hello"])
    client.create_index.assert_not_called()


def test_ensure_index_rejects_missing_embedding_aspect() -> None:
    client = make_client()
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(data=make_index(aspects=[]))

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"])


def test_ensure_index_rejects_dims_mismatch() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 999}
    )
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(data=make_index(aspects=[embedding_aspect]))

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"])


def test_ensure_index_rejects_missing_schema_aspect() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(data=make_index(aspects=[embedding_aspect]))

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        schema=[AspectSchema(name="source", type="keyword")],
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"], metadatas=[{"source": "wiki"}])


def test_ensure_index_rejects_schema_type_mismatch() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    source_aspect = Aspect(name="source", type="number", path="$.source", settings={})
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(
        data=make_index(aspects=[embedding_aspect, source_aspect])
    )

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
        schema=[AspectSchema(name="source", type="keyword")],
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"], metadatas=[{"source": "wiki"}])


def test_ensure_index_rejects_payload_key_collision() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    payload_aspect = Aspect(name="__payload", type="keyword", path="$.__payload", settings={})
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(
        data=make_index(aspects=[embedding_aspect, payload_aspect])
    )

    store = AspectedVectorStore(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )
    with pytest.raises(ValueError):
        store.add_texts(["hello"])


# ---------------------------------------------------------------------------
# from_existing_index
# ---------------------------------------------------------------------------


def test_from_existing_index_populates_settings() -> None:
    client = make_client()
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )
    source_aspect = Aspect(name="source", type="keyword", path="$.source", settings={"foo": "bar"})
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(
        data=make_index(
            distance_type=DistanceType.euclidean,
            aspects=[embedding_aspect, source_aspect],
            id_size=16,
        )
    )

    store = AspectedVectorStore.from_existing_index(
        client=client,
        embedding=FakeEmbeddings(),
        index_name="test-index",
    )

    assert store._distance_type == DistanceType.euclidean
    assert store._content_embedding_key == "__embedding"
    assert store._id_size == 16
    assert len(store._schema) == 1
    assert store._schema[0].name == "source"
    assert store._schema[0].type == "keyword"
    assert store._schema[0].settings == {"foo": "bar"}


def test_from_existing_index_missing_embedding_aspect_raises() -> None:
    client = make_client()
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(data=make_index(aspects=[]))

    with pytest.raises(ValueError):
        AspectedVectorStore.from_existing_index(
            client=client,
            embedding=FakeEmbeddings(),
            index_name="test-index",
        )


def test_from_existing_index_non_vector_embedding_aspect_raises() -> None:
    client = make_client()
    bad_aspect = Aspect(name="__embedding", type="keyword", path="$.__embedding", settings={})
    client.get_index.side_effect = None
    client.get_index.return_value = MagicMock(data=make_index(aspects=[bad_aspect]))

    with pytest.raises(ValueError):
        AspectedVectorStore.from_existing_index(
            client=client,
            embedding=FakeEmbeddings(),
            index_name="test-index",
        )


def test_from_existing_index_constructs_client_from_url() -> None:
    embedding_aspect = Aspect(
        name="__embedding", type="raw", path="$.__embedding", settings={"numDims": 4}
    )

    with patch("langchain_aspected.vectorstores.AspectedClient") as mock_client_cls:
        mock_client = make_client()
        mock_client.get_index.side_effect = None
        mock_client.get_index.return_value = MagicMock(data=make_index(aspects=[embedding_aspect]))
        mock_client_cls.return_value = mock_client

        store = AspectedVectorStore.from_existing_index(
            embedding=FakeEmbeddings(),
            index_name="test-index",
            url="http://example.com:1234",
        )

        mock_client_cls.assert_called_once_with(url="http://example.com:1234")
        assert store._client is mock_client
