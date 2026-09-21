"""LangChain VectorStore integration for the Aspected vector database."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from aspected_client import (
    Aspect,
    AspectedClient,
    CreateIndexPayload,
    DeleteOperation,
    DeleteOperationPolicy,
    DistanceType,
    Duplicate,
    Index,
    Missing,
    QuerySearch,
    QueryVector,
    SearchResultRow,
    UploadDocsPayload,
    UploadDocsPayloadPolicy,
    UpsertOperation,
    UpsertOperationPolicy,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Default field names used inside the Aspected document payload
_DEFAULT_CONTENT_KEY = "__payload"
_DEFAULT_CONTENT_EMBEDDING_KEY = "__embedding"
_DEFAULT_ID_SIZE = 36


@dataclass
class AspectSchema:
    """User-facing description of an extra Aspected aspect/column.

    Parameters
    ----------
    name:
        Name of the aspect/column.  Must match a key present in every
        document's metadata when indexing.
    type:
        Aspected aspect type (e.g. ``"keyword"``, ``"number"``, ``"raw"``).
    settings:
        Optional aspect-specific settings dict.  Defaults to an empty dict.
    """

    name: str
    type: str
    settings: dict[str, Any] | None = None

    def to_aspect(self) -> Aspect:
        """Convert to the underlying :class:`~aspected_client.Aspect` model."""
        return Aspect(
            name=self.name,
            type=self.type,
            path=f"$.{self.name}",
            settings=self.settings or {},
        )


class AspectedVectorStore(VectorStore):
    """LangChain ``VectorStore`` backed by the `Aspected <https://aspected.com>`_ vector database.

    Aspected is a multi-aspect vector database that supports sparse and dense
    retrieval over composite embeddings.  This integration uses Aspected's
    **vector** aspect type to store pre-computed LangChain embeddings inside
    each document, then runs k-NN searches via the ``$raw`` query mechanism.

    Parameters
    ----------
    client:
        An already-constructed :class:`~aspected_client.AspectedClient`.
    embedding:
        Any LangChain :class:`~langchain_core.embeddings.Embeddings` implementation
        used to embed query strings.
    index_name:
        Name of the Aspected index to read from / write to.
    content_payload_key:
        Key inside the Aspected document ``doc`` dict where the raw text is stored.
        Defaults to ``"__payload"``.
    content_embedding_key:
        Name of the Aspected aspect that holds the embedding vector.
        Defaults to ``"__embedding"``.
    distance_type:
        Distance metric used when creating a new index.
        Defaults to :attr:`~aspected_client.DistanceType.cosine`.
    id_size:
        Size (in bytes) of document IDs used when creating a new index.
        Defaults to ``36`` (the length of a UUID string).
    schema:
        Optional list of :class:`AspectSchema` describing additional aspects
        (columns) beyond the embedding vector.  Every document indexed via
        :meth:`add_texts` must supply a value for each of these fields in its
        metadata dict -- they are required, not optional.  Field names must
        not collide with ``content_payload_key`` or ``content_embedding_key``.

    Examples
    --------
    .. code-block:: python

        from aspected_client import AspectedClient
        from langchain_openai import OpenAIEmbeddings
        from langchain_aspected import AspectedVectorStore

        client = AspectedClient(url="http://localhost:8080")
        embeddings = OpenAIEmbeddings()

        store = AspectedVectorStore.from_texts(
            texts=["Hello world", "Foo bar"],
            embedding=embeddings,
            client=client,
            index_name="my-index",
        )
        docs = store.similarity_search("hello", k=2)
    """

    def __init__(
        self,
        client: AspectedClient,
        embedding: Embeddings,
        index_name: str,
        *,
        content_payload_key: str = _DEFAULT_CONTENT_KEY,
        content_embedding_key: str = _DEFAULT_CONTENT_EMBEDDING_KEY,
        distance_type: DistanceType = DistanceType.cosine,
        id_size: int = _DEFAULT_ID_SIZE,
        schema: list[AspectSchema] | None = None,
    ) -> None:
        self._client = client
        self._embedding = embedding
        self._index_name = index_name
        self._content_payload_key = content_payload_key
        self._content_embedding_key = content_embedding_key
        self._distance_type = distance_type
        self._id_size = id_size
        self._schema = list(schema) if schema else []
        self._validate_schema_names()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        """Expose the underlying embeddings model (required by LangChain)."""
        return self._embedding

    def _doc_to_langchain(self, row: SearchResultRow) -> Document:
        """Convert a :class:`~aspected_client.SearchResultRow` to a :class:`~langchain_core.documents.Document`."""
        doc = row.doc or {}
        page_content = str(doc.pop(self._content_payload_key, ""))
        # Remove the stored embedding vector from metadata if present
        doc.pop(self._content_embedding_key, None)
        return Document(page_content=page_content, metadata=doc)

    def _validate_schema_names(self) -> None:
        """Ensure schema field names don't collide with reserved keys."""
        reserved = {
            self._content_payload_key: "content_payload_key",
            self._content_embedding_key: "content_embedding_key",
        }
        for aspect in self._schema:
            if aspect.name in reserved:
                raise ValueError(
                    f"Schema field name {aspect.name!r} collides with "
                    f"{reserved[aspect.name]} ({aspect.name!r}). Choose a "
                    "different schema field name or reserved key value."
                )
        if self._content_payload_key == self._content_embedding_key:
            raise ValueError(
                "content_payload_key and content_embedding_key must not be "
                f"the same value ({self._content_payload_key!r})."
            )

    def _ensure_index(self, num_dims: int) -> None:
        """Create the Aspected index if it does not already exist.

        When the index already exists, it is validated against this store's
        configuration (embedding/payload keys, distance type, and schema)
        instead of being silently reused.
        """
        try:
            result = self._client.get_index(self._index_name)
        except Exception:
            self._client.create_index(
                self._index_name,
                CreateIndexPayload(
                    id_size=self._id_size,
                    distance_type=self._distance_type,
                    schema_=[
                        Aspect(
                            name=self._content_embedding_key,
                            type="raw",
                            path=f"$.{self._content_embedding_key}",
                            settings={"numDims": num_dims},
                        ),
                        *[aspect.to_aspect() for aspect in self._schema],
                    ],
                ),
            )
        else:
            self._validate_existing_index(result.data, num_dims)

    def _validate_existing_index(self, index: Index, num_dims: int) -> None:
        """Validate that an already-existing index matches this store's configuration."""
        if index.distance_type != self._distance_type:
            raise ValueError(
                f"Index {self._index_name!r} uses distance type "
                f"{index.distance_type!r}, but this store is configured "
                f"for {self._distance_type!r}."
            )

        if index.id_size != self._id_size:
            raise ValueError(
                f"Index {self._index_name!r} uses id_size {index.id_size!r}, "
                f"but this store is configured for id_size "
                f"{self._id_size!r}."
            )

        aspects_by_name = {aspect.name: aspect for aspect in index.schema_}

        embedding_aspect = aspects_by_name.get(self._content_embedding_key)
        if embedding_aspect is None:
            raise ValueError(
                f"Index {self._index_name!r} has no aspect named "
                f"{self._content_embedding_key!r} (content_embedding_key). "
                f"Available aspects: {sorted(aspects_by_name)}"
            )
        if embedding_aspect.type != "raw":
            raise ValueError(
                f"Aspect {self._content_embedding_key!r} on index "
                f"{self._index_name!r} is not a raw (vector) aspect (found "
                f"type {embedding_aspect.type!r})."
            )
        existing_dims = embedding_aspect.settings.get("numDims")
        if existing_dims is not None and existing_dims != num_dims:
            raise ValueError(
                f"Index {self._index_name!r} embedding aspect "
                f"{self._content_embedding_key!r} expects vectors of size "
                f"{existing_dims}, but the embedding model produces vectors "
                f"of size {num_dims}."
            )

        if self._content_payload_key in aspects_by_name:
            raise ValueError(
                f"content_payload_key {self._content_payload_key!r} "
                f"collides with an existing aspect on index "
                f"{self._index_name!r}."
            )

        for schema_aspect in self._schema:
            existing = aspects_by_name.get(schema_aspect.name)
            if existing is None:
                raise ValueError(
                    f"Schema field {schema_aspect.name!r} is not defined as "
                    f"an aspect on index {self._index_name!r}. Available "
                    f"aspects: {sorted(aspects_by_name)}"
                )
            if existing.type != schema_aspect.type:
                raise ValueError(
                    f"Schema field {schema_aspect.name!r} has type "
                    f"{schema_aspect.type!r}, but index "
                    f"{self._index_name!r} defines it with type "
                    f"{existing.type!r}."
                )
            if schema_aspect.settings is not None and schema_aspect.settings != existing.settings:
                raise ValueError(
                    f"Schema field {schema_aspect.name!r} has settings "
                    f"{schema_aspect.settings!r}, but index "
                    f"{self._index_name!r} defines it with settings "
                    f"{existing.settings!r}."
                )

    def _validate_metadata(self, metadatas: list[dict]) -> None:
        """Ensure every metadata dict contains all schema-required fields."""
        if not self._schema:
            return
        required_fields = [aspect.name for aspect in self._schema]
        for meta in metadatas:
            missing = [field for field in required_fields if field not in meta]
            if missing:
                raise ValueError(
                    f"Missing required schema field(s) {missing} in document metadata: {meta}"
                )

    # ------------------------------------------------------------------
    # VectorStore abstract methods
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed ``texts`` and upsert them into the Aspected index.

        Parameters
        ----------
        texts:
            Iterable of strings to embed and store.
        metadatas:
            List of metadata dicts, one per text.  When a ``schema`` was
            provided at construction time, every metadata dict *must*
            contain a value for each schema field -- these fields are
            required, not optional.
        ids:
            Optional list of document IDs.  UUIDs are generated when omitted.

        Returns
        -------
        list[str]
            The document IDs of the inserted records.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        vectors = self._embedding.embed_documents(texts_list)
        self._ensure_index(len(vectors[0]))

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        self._validate_metadata(metadatas)

        operations: list[UpsertOperation | DeleteOperation] = []
        for doc_id, text, vector, meta in zip(ids, texts_list, vectors, metadatas, strict=True):
            doc: dict[str, Any] = {
                self._content_payload_key: text,
                self._content_embedding_key: vector,
                **meta,
            }
            operations.append(
                UpsertOperation(
                    id=doc_id,
                    doc=doc,
                    policy=UpsertOperationPolicy(duplicate=Duplicate.replace_),
                )
            )

        self._client.upload_docs(
            self._index_name,
            UploadDocsPayload(
                data=operations,
                policy=UploadDocsPayloadPolicy(missing=Missing.ignore),
            ),
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the ``k`` most similar documents to ``query`` text.

        Parameters
        ----------
        query:
            Natural-language query string.
        k:
            Number of results to return.
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the ``k`` most similar documents to the given embedding vector.

        Parameters
        ----------
        embedding:
            A pre-computed query vector with the same dimensionality as the index.
        k:
            Number of results to return.
        """
        docs_and_scores = self.similarity_search_by_vector_with_score(embedding, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the ``k`` most similar documents paired with their distance scores.

        Parameters
        ----------
        query:
            Natural-language query string.
        k:
            Number of results to return.

        Returns
        -------
        list[tuple[Document, float]]
            Each tuple is ``(document, distance)`` where a *lower* distance
            means a *closer* match (following Aspected conventions).
        """
        query_vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(query_vector, k=k, **kwargs)

    def similarity_search_by_vector_with_score(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the ``k`` most similar documents to the given vector, paired with scores.

        Parameters
        ----------
        embedding:
            A pre-computed query vector.
        k:
            Number of results to return.
        """
        result = self._client.search_index(
            self._index_name,
            QuerySearch(
                k=k,
                query={self._content_embedding_key: QueryVector(**{"$raw": embedding})},
                with_docs=True,
            ),
        )
        return [(self._doc_to_langchain(row), row.distance) for row in result.data]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by ID from the Aspected index.

        Parameters
        ----------
        ids:
            List of document IDs to remove.

        Returns
        -------
        bool | None
            ``True`` if the operation was submitted successfully.
        """
        if not ids:
            return True

        operations: list[UpsertOperation | DeleteOperation] = [
            DeleteOperation(
                id=doc_id,
                doc=None,
                policy=DeleteOperationPolicy(missing=Missing.ignore),
            )
            for doc_id in ids
        ]
        self._client.upload_docs(
            self._index_name,
            UploadDocsPayload(
                data=operations,
                policy=UploadDocsPayloadPolicy(missing=Missing.ignore),
            ),
        )
        return True

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        client: AspectedClient | None = None,
        url: str = "http://localhost:8080",
        index_name: str = "langchain",
        content_payload_key: str = _DEFAULT_CONTENT_KEY,
        content_embedding_key: str = _DEFAULT_CONTENT_EMBEDDING_KEY,
        distance_type: DistanceType = DistanceType.cosine,
        id_size: int = _DEFAULT_ID_SIZE,
        schema: list[AspectSchema] | None = None,
        **kwargs: Any,
    ) -> AspectedVectorStore:
        """Create an :class:`AspectedVectorStore`, embed ``texts``, and upsert them.

        Parameters
        ----------
        texts:
            List of strings to embed and store.
        embedding:
            LangChain embeddings model.
        metadatas:
            Optional metadata dicts aligned with ``texts``.  When ``schema``
            is provided, every metadata dict must contain a value for each
            schema field.
        ids:
            Optional document IDs.
        client:
            Pre-constructed :class:`~aspected_client.AspectedClient`.  When
            *None*, a new client is constructed using ``url``.
        url:
            Aspected server URL (used only when ``client`` is *None*).
        index_name:
            Name of the Aspected index to create / use.
        content_payload_key:
            Key inside each Aspected document for the raw text payload.
        content_embedding_key:
            Aspected aspect name used for the embedding vector.
        distance_type:
            Distance metric for the index.
        id_size:
            Size (in bytes) of document IDs used when creating a new index.
        schema:
            Optional list of :class:`AspectSchema` describing additional
            required aspects/columns beyond the embedding vector.

        Returns
        -------
        AspectedVectorStore
        """
        if client is None:
            client = AspectedClient(url=url)

        store = cls(
            client=client,
            embedding=embedding,
            index_name=index_name,
            content_payload_key=content_payload_key,
            content_embedding_key=content_embedding_key,
            distance_type=distance_type,
            id_size=id_size,
            schema=schema,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        ids: list[str] | None = None,
        client: AspectedClient | None = None,
        url: str = "http://localhost:8080",
        index_name: str = "langchain",
        content_payload_key: str = _DEFAULT_CONTENT_KEY,
        content_embedding_key: str = _DEFAULT_CONTENT_EMBEDDING_KEY,
        distance_type: DistanceType = DistanceType.cosine,
        id_size: int = _DEFAULT_ID_SIZE,
        schema: list[AspectSchema] | None = None,
        **kwargs: Any,
    ) -> AspectedVectorStore:
        """Create an :class:`AspectedVectorStore` from a list of :class:`~langchain_core.documents.Document` objects.

        Parameters
        ----------
        documents:
            List of LangChain documents.  When ``schema`` is provided, every
            document's metadata must contain a value for each schema field.
        embedding:
            LangChain embeddings model.
        ids:
            Optional document IDs.
        client:
            Pre-constructed :class:`~aspected_client.AspectedClient`.
        url:
            Aspected server URL.
        index_name:
            Name of the Aspected index.
        content_payload_key:
            Doc field key for text storage.
        content_embedding_key:
            Aspected aspect name for the embedding.
        distance_type:
            Distance metric for the index.
        id_size:
            Size (in bytes) of document IDs used when creating a new index.
        schema:
            Optional list of :class:`AspectSchema` describing additional
            required aspects/columns beyond the embedding vector.

        Returns
        -------
        AspectedVectorStore
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            client=client,
            url=url,
            index_name=index_name,
            content_payload_key=content_payload_key,
            content_embedding_key=content_embedding_key,
            distance_type=distance_type,
            id_size=id_size,
            schema=schema,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Index management helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_existing_index(
        embedding: Embeddings,
        index_name: str,
        *,
        client: AspectedClient | None = None,
        url: str = "http://localhost:8080",
        content_payload_key: str = _DEFAULT_CONTENT_KEY,
        content_embedding_key: str = _DEFAULT_CONTENT_EMBEDDING_KEY,
    ) -> AspectedVectorStore:
        """Build an :class:`AspectedVectorStore` wrapping an index that already exists.

        Unlike :meth:`from_texts`/:meth:`from_documents`, this does **not**
        create or modify anything on the server.  All settings -- distance
        type, embedding dimensionality, and metadata schema -- are read
        directly from the result of ``client.get_index(index_name)``.

        Parameters
        ----------
        embedding:
            LangChain embeddings model.
        index_name:
            Name of the pre-existing Aspected index.
        client:
            Pre-constructed :class:`~aspected_client.AspectedClient`.  When
            *None*, a new client is constructed using ``url``.
        url:
            Aspected server URL (used only when ``client`` is *None*).
        content_payload_key:
            Key inside each Aspected document for the raw text payload.
            This is a plain document field (not an aspect), so it cannot be
            inferred from the index and must be supplied explicitly if it
            differs from the default.
        content_embedding_key:
            Name of the aspect on the existing index that holds the
            embedding vector.  Used to look up that aspect's settings
            (e.g. dimensionality) and to identify which aspect to exclude
            from the derived ``schema``.

        Returns
        -------
        AspectedVectorStore

        Raises
        ------
        ValueError
            If the index has no vector aspect named ``content_embedding_key``.
        """
        if client is None:
            client = AspectedClient(url=url)

        result = client.get_index(index_name)
        index = result.data

        aspects_by_name = {aspect.name: aspect for aspect in index.schema_}
        embedding_aspect = aspects_by_name.get(content_embedding_key)
        if embedding_aspect is None:
            raise ValueError(
                f"Index {index_name!r} has no aspect named "
                f"{content_embedding_key!r} (content_embedding_key). "
                f"Available aspects: {sorted(aspects_by_name)}"
            )
        if embedding_aspect.type != "raw":
            raise ValueError(
                f"Aspect {content_embedding_key!r} on index {index_name!r} "
                f"is not a raw (vector) aspect (found type "
                f"{embedding_aspect.type!r})."
            )

        schema = [
            AspectSchema(name=aspect.name, type=aspect.type, settings=aspect.settings)
            for aspect in index.schema_
            if aspect.name != content_embedding_key
        ]

        return AspectedVectorStore(
            client=client,
            embedding=embedding,
            index_name=index_name,
            content_payload_key=content_payload_key,
            content_embedding_key=content_embedding_key,
            distance_type=index.distance_type,
            id_size=index.id_size,
            schema=schema,
        )

    def delete_index(self) -> bool:
        """Delete the entire Aspected index (including all documents).

        Returns
        -------
        bool
            ``True`` if the index was deleted.
        """
        result = self._client.delete_index(self._index_name)
        return result.data.deleted
