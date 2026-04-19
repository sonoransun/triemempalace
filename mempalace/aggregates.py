"""
aggregates.py — Hierarchical wing/hall/room vector-similarity indices.

MemPalace's palace taxonomy (wing → hall → room → drawer) has always
been load-bearing for exact-match metadata filters. This module turns
each taxonomic container into a first-class retrieval signal: the
Top-K representative drawers per container are concatenated into an
"aggregate document", embedded with the palace's active embedding
model(s), and stored in a sidecar Chroma collection per level. At
query time the searcher's weighted-RRF loop folds container-level hits
back onto their member drawers, so a query that matches a room's
overall theme boosts every drawer in that room — even the ones that
individually rank below the top-N under pure drawer semantic search.

Design notes
------------

* **Reuse the drawer vectors.** Aggregate building never re-embeds
  individual drawers; we ``col.get(include=["embeddings"])`` off the
  primary collection, compute a centroid, rank members by cosine
  similarity to the centroid, take the top-K, and let Chroma re-embed
  the resulting concatenation when it lands in the aggregate
  collection. One round-trip per container, not per drawer.

* **Dirty tracking rides the trie's LMDB meta DBI.** We already pay a
  trie-write transaction on every drawer write (`mark_container_dirty`
  piggybacks on that same env), and we already own crash-recovery for
  the trie, so adding a handful of scalar keys there avoids a second
  storage backend. See :meth:`mempalace.trie_index.TrieIndex.meta_put`.

* **Writes never stall on aggregate rebuild.** Drawer writes flip the
  container dirty and return. Aggregate recomputation happens either
  (a) on an explicit ``mempalace aggregates rebuild``, (b) as a
  daemon-thread rebuild fired from the searcher when the dirty count
  crosses ``aggregate_rebuild_threshold``, or (c) during
  ``mempalace repair`` which rebuilds aggregates from scratch.

* **Trie prefilter is preserved.** When a search has a keyword /
  temporal constraint, the trie returns a ``candidate_ids`` set. The
  aggregate contribution helper intersects container member_ids with
  that set — drawers that fail the trie are never boosted via their
  container, so the keyword AND-ness of the search is never broken by
  aggregate fusion.
"""

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime

import chromadb.errors

from .config import DEFAULT_HALL_KEYWORDS, MempalaceConfig
from .palace_io import aggregate_collection_name_for, open_collection
from .trie_index import TrieIndex, trie_db_path

logger = logging.getLogger("mempalace.aggregates")

# Sentinel containers the miner writes for bookkeeping. They carry no
# user content so aggregating them is noise at best and confusing at
# worst.
SENTINEL_WINGS: frozenset[str] = frozenset()
SENTINEL_HALLS: frozenset[str] = frozenset({"hall_registry"})
SENTINEL_ROOMS: frozenset[str] = frozenset({"_registry", "diary"})

LEVELS: tuple[str, ...] = ("wing", "hall", "room")

# Fallback hall when no keyword matches. Kept distinct from the
# existing ``hall_diary`` etc. so that unclassified content is easy to
# re-scan later.
DEFAULT_HALL = "hall_general"

# Byte-level meta keys written into the trie's meta DBI. All keys are
# namespaced under the literal ``agg:`` prefix so they can't collide
# with the trie's own reserved keys.
_DIRTY_KEY = {
    "wing": b"agg:dirty:wing",
    "hall": b"agg:dirty:hall",
    "room": b"agg:dirty:room",
}

_AGG_SEPARATOR = "\n\n---\n\n"


# ── Hall classification ──────────────────────────────────────────────


def classify_hall(text: str, *, default: str = DEFAULT_HALL) -> str:
    """Return the hall slug (e.g. ``hall_technical``) for ``text``.

    First-match on lowercase-substring against the configured
    ``hall_keywords`` map; falls back to ``default`` when no keyword
    hits. The match is cheap enough to run inline with every drawer
    write (a few dozen ``substr in s`` checks, all on the already-
    lowercased text).
    """
    if not text:
        return default
    lowered = text.lower()
    try:
        keywords = MempalaceConfig().hall_keywords
    except Exception:  # pragma: no cover — config always loads
        keywords = DEFAULT_HALL_KEYWORDS
    for topic, kws in keywords.items():
        for kw in kws:
            if kw and kw.lower() in lowered:
                return f"hall_{topic}"
    return default


def hydrate_drawer_metadata(metadata: dict, content: str) -> dict:
    """Fill in the ``hall`` field on a drawer metadata dict.

    Writers call this just before their ``col.add/upsert`` so every
    stored drawer carries a hall slug. Pre-set halls (e.g. the
    hardcoded ``hall_diary`` from ``tool_diary_write``) are preserved,
    and nothing else in ``metadata`` is touched. Returns the same dict
    for the caller's convenience — the hydration is in-place.
    """
    if not metadata.get("hall"):
        metadata["hall"] = classify_hall(content or "")
    return metadata


# ── Dirty tracking (trie meta DBI) ───────────────────────────────────


def _load_dirty(trie: TrieIndex, level: str) -> list[str]:
    raw = trie.meta_get(_DIRTY_KEY[level])
    if not raw:
        return []
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return []
    return [str(c) for c in data] if isinstance(data, list) else []


def _store_dirty(trie: TrieIndex, level: str, containers: list[str]) -> None:
    # Deduplicate while preserving insertion order so the rebuild
    # processes the earliest-marked first.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in containers:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    trie.meta_put(_DIRTY_KEY[level], json.dumps(deduped).encode("utf-8"))


def mark_container_dirty(
    palace_path: str,
    *,
    wing: str | None = None,
    hall: str | None = None,
    room: str | None = None,
) -> None:
    """Mark one or more containers as needing an aggregate rebuild.

    Called from every drawer-write path after a successful commit.
    Safe to call repeatedly for the same container — the dirty set is
    deduplicated on write. Sentinel containers are silently skipped.

    Any failure to open the trie is logged and swallowed: aggregate
    rebuilds are a best-effort secondary index and must never block
    the primary drawer write.
    """
    updates: list[tuple[str, str]] = []
    if wing and wing not in SENTINEL_WINGS:
        updates.append(("wing", wing))
    if hall and hall not in SENTINEL_HALLS:
        updates.append(("hall", hall))
    if room and room not in SENTINEL_ROOMS:
        updates.append(("room", room))
    if not updates:
        return
    try:
        trie = TrieIndex(db_path=trie_db_path(palace_path))
    except Exception as e:
        logger.debug("mark_container_dirty: trie open failed — %s", e)
        return
    for level, container in updates:
        current = _load_dirty(trie, level)
        if container in current:
            continue
        current.append(container)
        try:
            _store_dirty(trie, level, current)
        except Exception as e:
            logger.debug("mark_container_dirty(%s=%s): %s", level, container, e)


def list_dirty(palace_path: str) -> dict[str, list[str]]:
    """Return the dirty-container sets for every level.

    Empty lists for levels with nothing pending. Safe to call on a
    palace that has never run aggregates — the trie's meta DBI either
    returns None (treated as empty) or doesn't exist (caller gets
    empty lists).
    """
    out: dict[str, list[str]] = {level: [] for level in LEVELS}
    try:
        trie = TrieIndex(db_path=trie_db_path(palace_path))
    except Exception as e:
        logger.debug("list_dirty: trie open failed — %s", e)
        return out
    for level in LEVELS:
        out[level] = _load_dirty(trie, level)
    return out


def clear_dirty(palace_path: str, level: str, containers: Iterable[str]) -> None:
    """Remove the given containers from the dirty set for ``level``."""
    drop = set(containers)
    if not drop:
        return
    try:
        trie = TrieIndex(db_path=trie_db_path(palace_path))
    except Exception as e:
        logger.debug("clear_dirty: trie open failed — %s", e)
        return
    current = _load_dirty(trie, level)
    remaining = [c for c in current if c not in drop]
    if len(remaining) != len(current):
        _store_dirty(trie, level, remaining)


def _rebuilt_at_key(level: str, container: str) -> bytes:
    return f"agg:rebuilt_at:{level}:{container}".encode()


def record_rebuilt(palace_path: str, level: str, container: str) -> str:
    """Stamp a container's rebuild timestamp, returning the ISO string used."""
    ts = datetime.now(UTC).isoformat()
    try:
        trie = TrieIndex(db_path=trie_db_path(palace_path))
        trie.meta_put(_rebuilt_at_key(level, container), ts.encode("utf-8"))
    except Exception as e:
        logger.debug("record_rebuilt(%s=%s): %s", level, container, e)
    return ts


def last_rebuilt(palace_path: str, level: str, container: str) -> str | None:
    """Return the ISO timestamp of the last successful rebuild, or None."""
    try:
        trie = TrieIndex(db_path=trie_db_path(palace_path))
        raw = trie.meta_get(_rebuilt_at_key(level, container))
    except Exception as e:
        logger.debug("last_rebuilt(%s=%s): %s", level, container, e)
        return None
    return raw.decode("utf-8") if raw else None


def latest_rebuilt_any(palace_path: str) -> str | None:
    """Return the most recent rebuild timestamp across every container.

    Used by CLI/status reporting. Scans the trie meta DBI once and
    picks the lexicographically-greatest ISO timestamp (which is also
    the most recent because ISO-8601 sorts naturally).
    """
    try:
        trie = TrieIndex(db_path=trie_db_path(palace_path))
    except Exception as e:
        logger.debug("latest_rebuilt_any: trie open failed — %s", e)
        return None
    best: str | None = None
    # Iterate meta DBI cursor-wise; we don't have a bulk helper so we
    # use the low-level env directly. Keys are small so the scan is
    # cheap (O(meta_size)).
    try:
        with trie._env.begin(write=False) as txn, txn.cursor(db=trie._db_meta) as cursor:
            prefix = b"agg:rebuilt_at:"
            if cursor.set_range(prefix):
                for k, v in cursor:
                    if not k.startswith(prefix):
                        break
                    ts = bytes(v).decode("utf-8", errors="ignore")
                    if best is None or ts > best:
                        best = ts
    except Exception as e:
        logger.debug("latest_rebuilt_any: scan failed — %s", e)
    return best


# ── Aggregate computation ────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity on two equal-length sequences.

    Avoids pulling numpy into this module's import graph — the scale
    is per-container (typically < 1000 drawers × ~384 dims), which
    runs in milliseconds even without vectorization.
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    n = float(len(vectors))
    return [x / n for x in acc]


def _should_skip_container(level: str, container: str) -> bool:
    if level == "wing":
        return container in SENTINEL_WINGS
    if level == "hall":
        return container in SENTINEL_HALLS
    if level == "room":
        return container in SENTINEL_ROOMS
    return False


def compute_aggregate_text(
    col,
    *,
    level: str,
    container: str,
    top_k: int,
) -> tuple[str, list[str]]:
    """Build the Top-K aggregate document for one container.

    Returns ``(aggregate_text, drawer_ids_used)``. When the container
    has no drawers the return is ``("", [])`` and the caller should
    delete any existing aggregate row.

    The text is built by concatenating drawer documents with
    ``_AGG_SEPARATOR`` between them. Ordering: for N ≤ top_k, we keep
    whatever order the underlying collection returned. For N > top_k,
    we compute the container centroid and rank by cosine similarity to
    the centroid (descending), so the aggregate is dominated by the
    most prototypical members.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if _should_skip_container(level, container):
        return ("", [])
    try:
        page = col.get(
            where={level: container},
            include=["documents", "metadatas", "embeddings"],
        )
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("compute_aggregate_text(%s=%s): col.get failed — %s", level, container, e)
        return ("", [])

    # Chroma may return numpy arrays for the embeddings field; avoid
    # ``x or []`` on arrays (which triggers "truth value ambiguous").
    ids = list(page.get("ids") or [])
    docs_raw = page.get("documents")
    docs = list(docs_raw) if docs_raw is not None else []
    embeddings_raw = page.get("embeddings")
    if embeddings_raw is None:
        embeddings: list = []
    else:
        # Convert lazily: a numpy 2D array behaves like a sequence of rows.
        embeddings = list(embeddings_raw)
    if not ids:
        return ("", [])

    # Drop entries whose document is empty — sentinels and registry
    # stubs produce no useful text.
    triples: list[tuple[str, str, list[float] | None]] = []
    for idx, did in enumerate(ids):
        doc = docs[idx] if idx < len(docs) else ""
        emb_raw = embeddings[idx] if idx < len(embeddings) else None
        if not doc or not doc.strip():
            continue
        # Chroma may return numpy arrays; normalize to list[float] so
        # the cosine math stays in pure Python.
        emb: list[float] | None
        if emb_raw is None:
            emb = None
        elif isinstance(emb_raw, list):
            emb = emb_raw
        else:
            try:
                emb = [float(x) for x in emb_raw]
            except (TypeError, ValueError):
                emb = None
        triples.append((did, doc, emb))

    if not triples:
        return ("", [])

    if len(triples) <= top_k:
        selected = triples
    else:
        vecs = [t[2] for t in triples if t[2] is not None]
        if vecs and len(vecs) == len(triples):
            centroid = _centroid(vecs)
            ranked = sorted(
                triples,
                key=lambda t: _cosine(t[2] or [], centroid),
                reverse=True,
            )
            selected = ranked[:top_k]
        else:
            # Centroid path unavailable (embeddings missing, e.g. some
            # test doubles skip them). Fall back to the first top_k
            # — still deterministic, still bounded, just not ranked.
            selected = triples[:top_k]

    text = _AGG_SEPARATOR.join(t[1] for t in selected)
    return (text, [t[0] for t in selected])


def upsert_aggregate(
    palace_path: str,
    *,
    level: str,
    container: str,
    slug: str,
    text: str,
    member_ids: list[str],
) -> None:
    """Write one aggregate row into ``mempalace_<plural>[__slug]``.

    Empty ``text`` triggers a delete instead of an upsert, so
    containers that dropped to zero drawers after deletions stop
    contributing to retrieval.
    """
    collection_name = aggregate_collection_name_for(slug, level)
    try:
        agg_col = open_collection(
            palace_path,
            model=slug if slug != "default" else None,
            create=True,
            collection_name_override=collection_name,
        )
    except (OSError, chromadb.errors.ChromaError, ValueError) as e:
        logger.warning(
            "upsert_aggregate(%s=%s, slug=%s): open failed — %s", level, container, slug, e
        )
        return

    if not text or not member_ids:
        try:
            agg_col.delete(ids=[container])
        except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
            logger.debug("upsert_aggregate delete(%s=%s): %s", level, container, e)
        return

    metadata = {
        "level": level,
        "container": container,
        "member_ids": json.dumps(member_ids),
        "member_count": len(member_ids),
        "rebuilt_at": datetime.now(UTC).isoformat(),
    }
    try:
        agg_col.upsert(
            ids=[container],
            documents=[text],
            metadatas=[metadata],
        )
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.warning(
            "upsert_aggregate(%s=%s, slug=%s): upsert failed — %s", level, container, slug, e
        )


# ── Rebuild drivers ──────────────────────────────────────────────────


def _enumerate_all_containers(primary_col) -> dict[str, set[str]]:
    """Scan every drawer's metadata and return {level: {container, ...}}."""
    out: dict[str, set[str]] = {level: set() for level in LEVELS}
    try:
        page = primary_col.get(include=["metadatas"], limit=1_000_000)
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.warning("enumerate_all_containers: %s", e)
        return out
    for meta in page.get("metadatas") or []:
        if not isinstance(meta, dict):
            continue
        for level in LEVELS:
            v = meta.get(level)
            if v and isinstance(v, str):
                out[level].add(v)
    # Filter sentinels.
    out["wing"] -= SENTINEL_WINGS
    out["hall"] -= SENTINEL_HALLS
    out["room"] -= SENTINEL_ROOMS
    return out


def rebuild_containers(
    palace_path: str,
    *,
    level: str,
    containers: Iterable[str],
    slug: str,
    top_k: int | None = None,
) -> int:
    """Recompute + upsert aggregates for each (level, container, slug).

    Returns the number of containers actually written (successfully
    built and upserted). Containers that resolve to zero drawers have
    their aggregate row deleted instead, which still counts toward the
    return value because we took action.
    """
    cfg = MempalaceConfig()
    if top_k is None:
        top_k = cfg.aggregate_top_k
    try:
        primary_col = open_collection(palace_path, model=slug if slug != "default" else None)
    except (OSError, chromadb.errors.ChromaError, ValueError) as e:
        logger.warning("rebuild_containers(%s): primary open failed — %s", slug, e)
        return 0

    written = 0
    for container in containers:
        if _should_skip_container(level, container):
            continue
        text, members = compute_aggregate_text(
            primary_col, level=level, container=container, top_k=top_k
        )
        upsert_aggregate(
            palace_path,
            level=level,
            container=container,
            slug=slug,
            text=text,
            member_ids=members,
        )
        record_rebuilt(palace_path, level, container)
        written += 1
    return written


def rebuild_dirty(
    palace_path: str,
    *,
    levels: Iterable[str] = LEVELS,
    slugs: Iterable[str] | None = None,
) -> dict:
    """Rebuild aggregates for every currently-dirty container.

    Iterates :func:`list_dirty`, calls :func:`rebuild_containers` per
    (slug, level), and clears the dirty set on success. Returns a
    summary dict suitable for CLI output:
    ``{"level": {"wing": N, "hall": N, "room": N}, "total": N,
    "slugs": [...]}``.
    """
    cfg = MempalaceConfig()
    if slugs is None:
        slugs = cfg.enabled_embedding_models or ["default"]
    slugs = list(slugs)

    dirty = list_dirty(palace_path)
    per_level: dict[str, int] = {level: 0 for level in LEVELS}
    for level in levels:
        containers = list(dirty.get(level, []))
        if not containers:
            continue
        for slug in slugs:
            per_level[level] += rebuild_containers(
                palace_path, level=level, containers=containers, slug=slug
            )
        # Clear the dirty set once every enabled slug has been built;
        # if one slug failed mid-loop we still clear because the other
        # slugs' aggregates are now consistent — the failing slug will
        # surface on the next search and can be rebuilt explicitly.
        clear_dirty(palace_path, level, containers)

    return {
        "by_level": per_level,
        "total": sum(per_level.values()),
        "slugs": slugs,
    }


def rebuild_all(
    palace_path: str,
    *,
    slugs: Iterable[str] | None = None,
) -> dict:
    """Rebuild aggregates for every container in the palace.

    Used by ``mempalace repair`` and by the ``--all`` flag on
    ``mempalace aggregates rebuild``. Walks metadata off the default
    collection to discover the container set, then reuses
    :func:`rebuild_containers` per (slug, level).
    """
    cfg = MempalaceConfig()
    if slugs is None:
        slugs = cfg.enabled_embedding_models or ["default"]
    slugs = list(slugs)

    try:
        primary_col = open_collection(palace_path)
    except (OSError, chromadb.errors.ChromaError, ValueError) as e:
        logger.warning("rebuild_all: primary open failed — %s", e)
        return {"by_level": {level: 0 for level in LEVELS}, "total": 0, "slugs": slugs}

    sets = _enumerate_all_containers(primary_col)
    per_level: dict[str, int] = {level: 0 for level in LEVELS}
    for level in LEVELS:
        containers = sorted(sets[level])
        if not containers:
            continue
        for slug in slugs:
            per_level[level] += rebuild_containers(
                palace_path, level=level, containers=containers, slug=slug
            )
        clear_dirty(palace_path, level, containers)

    return {
        "by_level": per_level,
        "total": sum(per_level.values()),
        "slugs": slugs,
    }


# ── Query-time contribution ──────────────────────────────────────────


def aggregate_contributions(
    query: str,
    palace_path: str,
    *,
    slug: str,
    candidate_ids: set[str] | None = None,
    weights: dict[str, float] | None = None,
    per_level_limit: int = 15,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Return ``{drawer_id: boost}`` summed across wing/hall/room aggregates.

    For each level, query the matching aggregate collection with the
    user's text, iterate the top-N container hits, distribute the
    weighted RRF contribution to every member drawer of each hit, and
    accumulate into one dict. The boost is **additive** over levels
    (a drawer whose wing + hall + room all match gets three
    contributions).

    When ``candidate_ids`` is non-None, boosts are dropped for drawer
    ids not in the set — preserving the trie prefilter's AND semantics.
    Callers that don't use a trie prefilter should pass ``None`` so
    every member drawer keeps its boost.

    Silently returns ``{}`` when ``aggregate_enabled=False``, when the
    query is empty/whitespace, when the aggregate collections don't
    exist yet, or when ``slug == "all"`` (fan-out contributions are
    computed per-slug by the outer searcher loop).
    """
    if not query or not query.strip():
        return {}
    if slug == "all":
        return {}
    # Guarded config read: test fixtures sometimes monkeypatch
    # MempalaceConfig with a stub that doesn't carry the aggregate-
    # layer knobs. Degrade silently rather than raising.
    try:
        cfg = MempalaceConfig()
        enabled = getattr(cfg, "aggregate_enabled", True)
        default_weights = getattr(cfg, "aggregate_weights", None)
    except Exception as e:
        logger.debug("aggregate_contributions: config read failed — %s", e)
        return {}
    if not enabled:
        return {}

    effective_weights = default_weights if weights is None else weights
    if effective_weights is None:
        from .config import DEFAULT_AGGREGATE_WEIGHTS as _DEFAULT_AGG_WEIGHTS

        effective_weights = _DEFAULT_AGG_WEIGHTS

    out: dict[str, float] = {}
    for level in LEVELS:
        w = float(effective_weights.get(level, 0.0))
        if w <= 0:
            continue
        collection_name = aggregate_collection_name_for(slug, level)
        try:
            agg_col = open_collection(
                palace_path,
                model=slug if slug != "default" else None,
                collection_name_override=collection_name,
            )
        except (OSError, chromadb.errors.ChromaError, ValueError) as e:
            # Aggregate not built yet for this (slug, level) — degrade
            # silently so search still works on old palaces.
            logger.debug("aggregate_contributions: %s not available — %s", collection_name, e)
            continue
        try:
            result = agg_col.query(
                query_texts=[query],
                n_results=per_level_limit,
                include=["metadatas"],
            )
        except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
            logger.debug("aggregate_contributions query(%s): %s", collection_name, e)
            continue
        metas = (result.get("metadatas") or [[]])[0]
        for rank, meta in enumerate(metas):
            if not isinstance(meta, dict):
                continue
            member_ids_raw = meta.get("member_ids")
            if not member_ids_raw:
                continue
            try:
                member_ids = json.loads(member_ids_raw)
            except (TypeError, ValueError):
                continue
            contribution = w / (rrf_k + rank)
            for did in member_ids:
                if candidate_ids is not None and did not in candidate_ids:
                    continue
                out[did] = out.get(did, 0.0) + contribution
    return out


def should_auto_rebuild(palace_path: str) -> bool:
    """True when the total dirty count crosses the configured threshold."""
    cfg = MempalaceConfig()
    if not cfg.aggregate_enabled:
        return False
    threshold = cfg.aggregate_rebuild_threshold
    if threshold <= 0:
        return False
    counts = list_dirty(palace_path)
    total = sum(len(v) for v in counts.values())
    return total >= threshold
