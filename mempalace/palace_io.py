"""
palace_io.py — Single seam for opening ChromaDB collections.

Before this module existed, every caller (miner, convo_miner, searcher,
layers, palace_graph, mcp_server, cli) duplicated the same four-line
incantation:

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection("mempalace_drawers")  # or get_or_create

With configurable embedding models, that pattern stops working — the
collection name depends on the model slug, and the embedding function
has to be passed on create. Rather than sprinkle that logic across eight
files we funnel every open through :func:`open_collection`.

What this module owns
---------------------

* Resolving a (possibly-None) ``model`` argument to the palace's default
  via ``MempalaceConfig``.
* Translating slug → collection name via
  :func:`mempalace.embeddings.collection_name_for`.
* Lazy-loading the embedding function via
  :func:`mempalace.embeddings.load_embedding_function` (returns ``None``
  for the ``default`` slug so Chroma uses its built-in ONNX mini-lm).
* Caching Chroma ``PersistentClient`` instances per canonical palace
  path — Chroma itself is happy to be re-opened but the handshake
  overhead is ~5 ms and we call it many times from the MCP hot path.
* Caching collection handles per (client, collection_name) pair.

What this module does **not** own
----------------------------------

* ChromaDB version quirks — we assume 0.5.x / 0.6.x.
* Multi-process coordination — MemPalace is single-process, and Chroma
  handles writer concurrency via its SQLite-backed metadata store.
* Schema migration. The trie index has its own migration; Chroma
  collections are migrated by ``mempalace repair``.
"""

import os
import threading
from pathlib import Path
from typing import Any

import chromadb

from . import embeddings as _embeddings

# Process-wide caches. Guarded by a single lock because the MCP server
# may be multi-threaded in future (py-lmdb forbids it today but we
# don't want cache corruption if that changes).
_client_cache: dict[str, chromadb.PersistentClient] = {}
_collection_cache: dict[tuple[str, str], Any] = {}
_lock = threading.Lock()


_AGGREGATE_LEVELS = ("wing", "hall", "room")
_AGGREGATE_LEVEL_PLURAL = {"wing": "wings", "hall": "halls", "room": "rooms"}


def aggregate_collection_name_for(slug: str, level: str) -> str:
    """Return the sidecar collection name for a (model-slug, level) pair.

    Mirrors :func:`mempalace.embeddings.collection_name_for` but targets
    the wing/hall/room aggregate indices built by
    :mod:`mempalace.aggregates`.

    Examples::

        aggregate_collection_name_for("default", "room") == "mempalace_rooms"
        aggregate_collection_name_for("bge-small-en", "hall") == "mempalace_halls__bge_small_en"
    """
    if level not in _AGGREGATE_LEVEL_PLURAL:
        raise ValueError(f"aggregate level must be one of {_AGGREGATE_LEVELS}, got {level!r}")
    base = f"mempalace_{_AGGREGATE_LEVEL_PLURAL[level]}"
    if slug == "default":
        return base
    return base + "__" + _embeddings.normalize_slug_for_collection(slug)


def _resolve_model(model: str | None) -> str:
    """Return the effective model slug.

    ``None`` falls back to ``config.default_embedding_model``. Import
    happens inside the function to avoid a module-load-time dependency
    on config (which in turn touches the filesystem).
    """
    if model is not None:
        return model
    from .config import MempalaceConfig

    return MempalaceConfig().default_embedding_model


def _canonical(palace_path: str | os.PathLike[str]) -> str:
    """Expand ``~`` and make absolute without following symlinks.

    We use ``absolute()`` rather than ``resolve()`` so that callers who
    intentionally use a symlinked palace directory keep their symlink
    intact in the cache key.
    """
    return str(Path(palace_path).expanduser().absolute())


def _get_client(palace_path: str | os.PathLike[str]) -> chromadb.PersistentClient:
    """Return a cached ``PersistentClient`` for the given palace path."""
    canonical = _canonical(palace_path)
    with _lock:
        client = _client_cache.get(canonical)
        if client is None:
            Path(canonical).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=canonical)
            _client_cache[canonical] = client
    return client


def open_collection(
    palace_path: str,
    *,
    model: str | None = None,
    create: bool = False,
    collection_name_override: str | None = None,
) -> Any:
    """Open the Chroma collection bound to ``model`` under ``palace_path``.

    Parameters
    ----------
    palace_path:
        Path to the palace directory. Created if missing.
    model:
        Embedding model slug. ``None`` means "use the palace's default
        model" from :class:`mempalace.config.MempalaceConfig`. Ignored
        when ``collection_name_override`` is set.
    create:
        If ``True``, create the collection if it doesn't exist. If
        ``False`` (the default), raise whatever Chroma raises on a
        missing collection.
    collection_name_override:
        Bypass the model→name lookup entirely and open the named
        collection directly using Chroma's built-in embedding. The only
        sanctioned caller is ``mempalace compress`` which writes to the
        ``mempalace_compressed`` sidecar collection that isn't part of
        the embedding registry.

    Returns
    -------
    chromadb Collection
        Cached per (palace_path, collection_name). The same underlying
        object is returned on every call until :func:`close_all` clears
        the cache.

    Behavior notes
    --------------
    * For ``model="default"`` the caller still gets a fully-functional
      collection bound to Chroma's built-in ONNX mini-lm — we pass no
      ``embedding_function=`` kwarg, matching the pre-multi-model
      behavior byte-for-byte.
    * For non-default slugs we load the embedding function via the
      registry. If the required optional extras aren't installed the
      registry raises with a "pip install mempalace[extras-name]" hint.
    * Slug ``"all"`` is **not** valid here — that's a query-time fan-out
      mode handled by the searcher, not a real collection. Passing it
      raises ``ValueError``.
    """
    if collection_name_override is not None:
        client = _get_client(palace_path)
        canonical_palace = _canonical(palace_path)
        collection_name = collection_name_override
        cache_key = (canonical_palace, collection_name)
        with _lock:
            cached = _collection_cache.get(cache_key)
            if cached is not None:
                return cached
        if create:
            col = client.get_or_create_collection(collection_name)
        else:
            col = client.get_collection(collection_name)
        with _lock:
            _collection_cache[cache_key] = col
        return col

    slug = _resolve_model(model)
    if slug == "all":
        raise ValueError(
            "open_collection: model='all' is a query-time fan-out mode, "
            "not a real collection. Pick a concrete slug for writes."
        )

    client = _get_client(palace_path)
    canonical_palace = _canonical(palace_path)
    collection_name = _embeddings.collection_name_for(slug)
    cache_key = (canonical_palace, collection_name)

    with _lock:
        cached = _collection_cache.get(cache_key)
        if cached is not None:
            return cached

    # Load the embedding function. Returns None for the default slug —
    # in that case we must NOT pass embedding_function= to Chroma, or
    # Chroma will think we're asking for an explicit override.
    ef = _embeddings.load_embedding_function(slug)

    # HNSW tuning. Chroma reads ``hnsw:*`` metadata keys at collection
    # creation time. For existing collections Chroma ignores the
    # metadata arg on ``get_or_create_collection`` (it won't overwrite
    # stored settings), so new collections pick up our tuned
    # ``hnsw:search_ef`` and existing palaces keep whatever they were
    # built with. Users who want to bump the value on an existing
    # palace can ``mempalace repair``.
    #
    # Defensive ``getattr``: tests monkeypatch MempalaceConfig with a
    # minimal ``_Stub`` that may not carry the new attribute.
    from .config import DEFAULT_HNSW_EF_SEARCH, MempalaceConfig

    hnsw_metadata = {
        "hnsw:search_ef": getattr(MempalaceConfig(), "hnsw_ef_search", DEFAULT_HNSW_EF_SEARCH),
    }

    if create:
        if ef is None:
            col = client.get_or_create_collection(collection_name, metadata=hnsw_metadata)
        else:
            col = client.get_or_create_collection(
                collection_name, embedding_function=ef, metadata=hnsw_metadata
            )
    else:
        if ef is None:
            col = client.get_collection(collection_name)
        else:
            col = client.get_collection(collection_name, embedding_function=ef)

    with _lock:
        _collection_cache[cache_key] = col
    return col


def delete_collection(
    palace_path: str,
    *,
    model: str | None = None,
    collection_name_override: str | None = None,
) -> None:
    """Drop a collection by name from the underlying Chroma client.

    Used by ``mempalace repair`` between the read-old / create-new
    cycle. Resolves the same (model | override) → name mapping as
    :func:`open_collection` and invalidates the cache for the dropped
    name so the next ``open_collection`` returns a fresh handle.
    """
    client = _get_client(palace_path)
    canonical_palace = _canonical(palace_path)
    if collection_name_override is not None:
        collection_name = collection_name_override
    else:
        slug = _resolve_model(model)
        if slug == "all":
            raise ValueError(
                "delete_collection: model='all' is a query-time fan-out mode, "
                "not a real collection."
            )
        collection_name = _embeddings.collection_name_for(slug)
    client.delete_collection(collection_name)
    with _lock:
        _collection_cache.pop((canonical_palace, collection_name), None)


def close_all() -> None:
    """Drop every cached client and collection handle.

    Used by tests between temp palaces, and by ``TrieIndex.close``-style
    shutdown flows. Does not delete data from disk.
    """
    with _lock:
        _collection_cache.clear()
        _client_cache.clear()


def drop_collection_cache(palace_path: str | os.PathLike[str]) -> None:
    """Invalidate cached collection handles for one palace.

    Called from ``cmd_repair`` after ``delete_collection`` so the next
    caller gets a fresh handle.
    """
    canonical = _canonical(palace_path)
    with _lock:
        _collection_cache.clear()  # conservative — rarely called
        _client_cache.pop(canonical, None)
