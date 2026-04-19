#!/usr/bin/env python3
"""
MemPalace MCP Server — read/write palace access for Claude Code
================================================================
Install: claude mcp add mempalace -- python -m mempalace.mcp_server [--palace /path/to/palace]

Tools (read):
  mempalace_status          — total drawers, wing/room breakdown
  mempalace_list_wings      — all wings with drawer counts
  mempalace_list_rooms      — rooms within a wing
  mempalace_get_taxonomy    — full wing → room → count tree
  mempalace_search          — semantic search, optional wing/room filter
  mempalace_check_duplicate — check if content already exists before filing

Tools (write):
  mempalace_add_drawer      — file verbatim content into a wing/room
  mempalace_delete_drawer   — remove a drawer by ID

Tools (maintenance):
  mempalace_reconnect       — force cache invalidation and reconnect after external writes
"""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import chromadb.errors
import lmdb

from . import embeddings as _embeddings
from .config import MempalaceConfig, sanitize_content, sanitize_name
from .knowledge_graph import KnowledgeGraph
from .palace_graph import (
    create_tunnel,
    delete_tunnel,
    find_tunnels,
    follow_tunnels,
    graph_stats,
    list_tunnels,
    traverse,
)
from .palace_io import open_collection
from .searcher import hybrid_search, search_memories
from .trie_index import TrieIndex, trie_db_path
from .version import __version__

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("mempalace_mcp")


def _parse_args():
    parser = argparse.ArgumentParser(description="MemPalace MCP Server")
    parser.add_argument(
        "--palace",
        metavar="PATH",
        help="Path to the palace directory (overrides config file and env var)",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("Ignoring unknown args: %s", unknown)
    return args


_args = _parse_args()

if _args.palace:
    os.environ["MEMPALACE_PALACE_PATH"] = str(Path(_args.palace).expanduser().absolute())

_config = MempalaceConfig()
# Only override KG path when --palace is explicitly provided; otherwise use
# KnowledgeGraph's default (~/.mempalace/knowledge_graph.sqlite3).
if _args.palace:
    _kg = KnowledgeGraph(db_path=str(Path(_config.palace_path) / "knowledge_graph.sqlite3"))
else:
    _kg = KnowledgeGraph()


# Per-process caches. _collection_cache is now a dict keyed on the
# resolved model slug so tools can switch collections per call without
# re-opening. Cleared by tests via `_reset_mcp_cache` in conftest.py.
_client_cache = None
_collection_cache: dict[str, object] = {}
_trie_cache = None


def _get_collection(create=False, *, model=None):
    """Return the ChromaDB collection bound to ``model`` (defaulting to
    the palace's configured default model).

    Caches one collection handle per (palace_path, model_slug) so
    repeat calls from the MCP hot path don't re-open Chroma.
    """
    global _collection_cache
    slug = model if model is not None else _config.default_embedding_model
    cached = _collection_cache.get(slug)
    if cached is not None:
        return cached
    try:
        col = open_collection(_config.palace_path, model=slug, create=create)
    except (OSError, chromadb.errors.ChromaError) as e:
        logger.debug("mcp: collection open failed (model=%s) — %s", slug, e)
        return None
    _collection_cache[slug] = col
    return col


# ==================== WRITE-AHEAD LOG ====================
# Every write operation is logged to a JSONL file before execution.
# This provides an audit trail for detecting memory poisoning and
# enables review/rollback of writes from external or untrusted sources.

_WAL_DIR = Path(os.path.expanduser("~/.mempalace/wal"))
_WAL_DIR.mkdir(parents=True, exist_ok=True)
# Windows doesn't support Unix permissions; chmod is a no-op there.
with contextlib.suppress(OSError, NotImplementedError):
    _WAL_DIR.chmod(0o700)
_WAL_FILE = _WAL_DIR / "write_log.jsonl"
# Pre-create WAL file with restricted permissions to avoid race condition
if not _WAL_FILE.exists():
    _WAL_FILE.touch(mode=0o600)
else:
    try:
        _WAL_FILE.chmod(0o600)
    except (OSError, NotImplementedError):
        pass

# Keys whose values should be redacted in WAL entries to avoid logging sensitive content
_WAL_REDACT_KEYS = frozenset(
    {"content", "content_preview", "document", "entry", "entry_preview", "query", "text"}
)


def _wal_log(operation: str, params: dict, result: dict | None = None) -> None:
    """Append a write operation to the write-ahead log."""
    # Redact sensitive content from params before logging
    safe_params = {}
    for k, v in params.items():
        if k in _WAL_REDACT_KEYS:
            safe_params[k] = f"[REDACTED {len(v)} chars]" if isinstance(v, str) else "[REDACTED]"
        else:
            safe_params[k] = v
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "params": safe_params,
        "result": result,
    }
    try:
        fd = os.open(str(_WAL_FILE), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        with contextlib.suppress(OSError, NotImplementedError):
            _WAL_FILE.chmod(0o600)
    except OSError as e:
        # Defensive: WAL writes are best-effort. The audit log must never
        # block the actual operation, so a failed open/write/chmod degrades
        # to an error log instead of propagating to the MCP caller.
        logger.error("WAL write failed: %s", e)


def _get_trie_index():
    """Return a cached ``TrieIndex`` rooted at the current palace path."""
    global _trie_cache
    if _trie_cache is None:
        _trie_cache = TrieIndex(db_path=trie_db_path(_config.palace_path))
    return _trie_cache


def _no_palace():
    return {
        "error": "No palace found",
        "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
    }


# ==================== HELPERS ====================


def _fetch_all_metadata(col, where=None):
    """Paginate col.get() to avoid the 10K silent truncation limit."""
    total = col.count()
    all_meta = []
    offset = 0
    while offset < total:
        kwargs = {"include": ["metadatas"], "limit": 1000, "offset": offset}
        if where:
            kwargs["where"] = where
        batch = col.get(**kwargs)
        if not batch["metadatas"]:
            break
        all_meta.extend(batch["metadatas"])
        offset += len(batch["metadatas"])
    return all_meta


_metadata_cache = None
_metadata_cache_time = 0
_METADATA_CACHE_TTL = 5.0  # seconds
_MAX_RESULTS = 100  # upper bound for search/list limit params


def _get_cached_metadata(col, where=None):
    """Return cached metadata if fresh, else fetch and cache."""
    global _metadata_cache, _metadata_cache_time
    now = time.time()
    if (
        where is None
        and _metadata_cache is not None
        and (now - _metadata_cache_time) < _METADATA_CACHE_TTL
    ):
        return _metadata_cache
    result = _fetch_all_metadata(col, where=where)
    if where is None:
        _metadata_cache = result
        _metadata_cache_time = now
    return result


def _sanitize_optional_name(value: str = None, field_name: str = "name") -> str:
    """Validate optional wing/room-style filters."""
    if value is None:
        return None
    return sanitize_name(value, field_name)


# ==================== READ TOOLS ====================


def tool_status() -> dict:
    col = _get_collection()
    if not col:
        return _no_palace()
    count = col.count()
    wings = {}
    rooms = {}
    try:
        all_meta = col.get(include=["metadatas"], limit=10000)["metadatas"]
        for m in all_meta:
            w = m.get("wing", "unknown")
            r = m.get("room", "unknown")
            wings[w] = wings.get(w, 0) + 1
            rooms[r] = rooms.get(r, 0) + 1
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("tool_status: metadata scan failed — %s", e)

    trie_stats = {}
    try:
        trie_stats = _get_trie_index().stats()
    except (lmdb.Error, OSError, RuntimeError) as e:
        logger.debug("tool_status: trie stats failed — %s", e)

    # Per-model drawer counts. Iterate every enabled model and count
    # drawers in its collection; skip silently if the collection doesn't
    # exist yet (user hasn't mined with that model).
    models_block = []
    for slug in _config.enabled_embedding_models:
        try:
            model_col = _get_collection(model=slug)
            models_block.append(
                {
                    "slug": slug,
                    "drawers": model_col.count() if model_col is not None else 0,
                    "default": slug == _config.default_embedding_model,
                }
            )
        except (OSError, chromadb.errors.ChromaError, ValueError) as e:
            logger.debug("tool_status: model %s count failed — %s", slug, e)
            models_block.append(
                {"slug": slug, "drawers": 0, "default": slug == _config.default_embedding_model}
            )

    # Aggregate-layer status: dirty-container counts per level and the
    # most recent rebuild timestamp. Cheap to compute (one meta DBI
    # read per level) and only costs ~100 µs when the palace has no
    # aggregate state yet.
    aggregates_block = {
        "enabled": _config.aggregate_enabled,
        "top_k": _config.aggregate_top_k,
        "weights": _config.aggregate_weights,
        "dirty": {"wing": 0, "hall": 0, "room": 0},
        "last_rebuild": None,
    }
    try:
        from . import aggregates as _agg_module

        dirty = _agg_module.list_dirty(_config.palace_path)
        aggregates_block["dirty"] = {
            level: len(dirty.get(level, [])) for level in _agg_module.LEVELS
        }
        aggregates_block["last_rebuild"] = _agg_module.latest_rebuilt_any(_config.palace_path)
    except Exception as e:
        logger.debug("tool_status: aggregates block failed — %s", e)

    return {
        "total_drawers": count,
        "wings": wings,
        "rooms": rooms,
        "palace_path": _config.palace_path,
        "trie": trie_stats,
        "models": models_block,
        "default_model": _config.default_embedding_model,
        "aggregates": aggregates_block,
        "protocol": PALACE_PROTOCOL,
        "aaak_dialect": AAAK_SPEC,
    }


def tool_list_models() -> dict:
    """Return the full embedding-model registry with install/enable flags.

    For each spec, report whether the required optional extras are
    installed in the current Python environment, whether the slug is
    in the palace's ``enabled_embedding_models`` list, and how many
    drawers live in its collection.
    """
    enabled = set(_config.enabled_embedding_models)
    default_slug = _config.default_embedding_model
    out = []
    for spec in _embeddings.list_specs():
        drawer_count = 0
        if spec.slug in enabled:
            try:
                col = _get_collection(model=spec.slug)
                drawer_count = col.count() if col is not None else 0
            except (OSError, chromadb.errors.ChromaError, ValueError) as e:
                logger.debug("tool_list_models: %s count failed — %s", spec.slug, e)
                drawer_count = 0
        out.append(
            {
                "slug": spec.slug,
                "display_name": spec.display_name,
                "description": spec.description,
                "backend": spec.backend,
                "model_id": spec.model_id,
                "dimension": spec.dimension,
                "context_tokens": spec.context_tokens,
                "extras_required": list(spec.extras_required),
                "installed": _embeddings.is_installed(spec),
                "enabled": spec.slug in enabled,
                "is_default": spec.slug == default_slug,
                "drawers": drawer_count,
                "supports_matryoshka": getattr(spec, "supports_matryoshka", False),
                "truncate_dim": getattr(spec, "truncate_dim", None),
            }
        )
    return {"models": out, "default_model": default_slug}


def tool_list_rerankers() -> dict:
    """Return the full reranker registry with install + pruning flags.

    Each entry reports whether the required optional extras are
    installed, whether the reranker supports per-token pruning
    (currently only Provence), and carries a human-readable
    description the AI can use to decide which reranker to pass as
    ``rerank=<slug>`` on subsequent search calls.

    Gracefully handles the case where neither ``rerank-provence`` nor
    ``rerank-bge`` is installed — returns the registry with
    ``installed=False`` for every spec so the AI knows what to suggest.
    """
    from . import rerank as _rerank_module

    out = []
    for spec in _rerank_module.list_reranker_specs():
        out.append(
            {
                "slug": spec.slug,
                "display_name": spec.display_name,
                "description": spec.description,
                "backend": spec.backend,
                "model_id": spec.model_id,
                "max_length": spec.max_length,
                "supports_pruning": spec.supports_pruning,
                "extras_required": list(spec.extras_required),
                "installed": _rerank_module.is_installed(spec),
            }
        )
    return {"rerankers": out}


# ── AAAK Dialect Spec ─────────────────────────────────────────────────────────
# Included in status response so the AI learns it on first wake-up call.
# Also available via mempalace_get_aaak_spec tool.

PALACE_PROTOCOL = """IMPORTANT — MemPalace Memory Protocol:
1. ON WAKE-UP: Call mempalace_status to load palace overview + AAAK spec.
2. BEFORE RESPONDING about any person, project, or past event: call mempalace_kg_query or mempalace_search FIRST. Never guess — verify.
3. IF UNSURE about a fact (name, gender, age, relationship): say "let me check" and query the palace. Wrong is worse than slow.
4. AFTER EACH SESSION: call mempalace_diary_write to record what happened, what you learned, what matters.
5. WHEN FACTS CHANGE: call mempalace_kg_invalidate on the old fact, mempalace_kg_add for the new one.

This protocol ensures the AI KNOWS before it speaks. Storage is not memory — but storage + this protocol = memory."""

AAAK_SPEC = """AAAK is a compressed memory dialect that MemPalace uses for efficient storage.
It is designed to be readable by both humans and LLMs without decoding.

FORMAT:
  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.
  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.
  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | ⚠: warnings/reminders.
  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g., 570x).
  IMPORTANCE: ★ to ★★★★★ (1-5 scale).
  HALLS: hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice.
  WINGS: wing_user, wing_agent, wing_team, wing_code, wing_myproject, wing_hardware, wing_ue5, wing_ai_research.
  ROOMS: Hyphenated slugs representing named ideas (e.g., chromadb-setup, gpu-pricing).

EXAMPLE:
  FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)

Read AAAK naturally — expand codes mentally, treat *markers* as emotional context.
When WRITING AAAK: use entity codes, mark emotions, keep structure tight."""


def tool_list_wings() -> dict:
    col = _get_collection()
    if not col:
        return _no_palace()
    wings = {}
    result = {"wings": wings}
    try:
        all_meta = _get_cached_metadata(col)
        for m in all_meta:
            w = m.get("wing", "unknown")
            wings[w] = wings.get(w, 0) + 1
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("tool_list_wings: metadata scan failed — %s", e)
    return {"wings": wings}


def tool_list_rooms(wing: str = None) -> dict:
    try:
        wing = _sanitize_optional_name(wing, "wing")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    if not col:
        return _no_palace()
    rooms = {}
    result = {"wing": wing or "all", "rooms": rooms}
    try:
        where = {"wing": wing} if wing else None
        all_meta = _fetch_all_metadata(col, where=where)
        for m in all_meta:
            r = m.get("room", "unknown")
            rooms[r] = rooms.get(r, 0) + 1
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("tool_list_rooms: metadata scan failed — %s", e)
    return {"wing": wing or "all", "rooms": rooms}


def tool_get_taxonomy() -> dict:
    col = _get_collection()
    if not col:
        return _no_palace()
    taxonomy = {}
    result = {"taxonomy": taxonomy}
    try:
        all_meta = _get_cached_metadata(col)
        for m in all_meta:
            w = m.get("wing", "unknown")
            r = m.get("room", "unknown")
            if w not in taxonomy:
                taxonomy[w] = {}
            taxonomy[w][r] = taxonomy[w].get(r, 0) + 1
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("tool_get_taxonomy: metadata scan failed — %s", e)
    return {"taxonomy": taxonomy}


def tool_search(
    query: str,
    limit: int = 5,
    wing: str = None,
    room: str = None,
    model: str = None,
    compress: str = "auto",
    token_budget: int = None,
    dup_threshold: float = 0.7,
    sent_threshold: float = 0.75,
    novelty_threshold: float = 0.2,
    rerank: str = None,
    rerank_prune: bool = True,
    kg_ppr: bool = False,
    aggregate_weights: dict = None,
):
    """Semantic search against a single model's collection.

    ``model`` unset = palace default. ``model="all"`` runs RRF fan-out
    across every enabled model and returns fused results. ``compress``
    selects the result-compression mode (auto/none/dedupe/sentences/
    aggressive); defaults to dedupe on fan-out and none otherwise.
    ``rerank`` (optional: "provence" / "bge") runs a cross-encoder
    reranker over the Chroma hits before compression — see
    ``mempalace_list_rerankers`` for install status. ``kg_ppr=True``
    turns on HippoRAG-style KG fusion.
    """
    try:
        wing = _sanitize_optional_name(wing, "wing")
        room = _sanitize_optional_name(room, "room")
    except ValueError as e:
        return {"error": str(e)}
    if model == "all" or rerank or kg_ppr or aggregate_weights is not None:
        # Any rerank/ppr/aggregate-weighting invocation goes through
        # hybrid_search because search_memories strips the extended
        # envelope fields (aggregate boosts live there).
        return hybrid_search(
            query,
            palace_path=_config.palace_path,
            wing=wing,
            room=room,
            n_results=limit,
            model=model,
            compress=compress,
            token_budget=token_budget,
            dup_threshold=dup_threshold,
            sent_threshold=sent_threshold,
            novelty_threshold=novelty_threshold,
            rerank=rerank,
            rerank_prune=rerank_prune,
            enable_kg_ppr=kg_ppr,
            aggregate_weights=aggregate_weights,
        )
    return search_memories(
        query,
        palace_path=_config.palace_path,
        wing=wing,
        room=room,
        n_results=limit,
        model=model,
        compress=compress,
        token_budget=token_budget,
    )


def tool_hybrid_search(
    query: str = "",
    keywords: list = None,
    keyword_mode: str = "all",
    since: str = None,
    until: str = None,
    as_of: str = None,
    wing: str = None,
    room: str = None,
    limit: int = 5,
    model: str = None,
    compress: str = "auto",
    token_budget: int = None,
    dup_threshold: float = 0.7,
    sent_threshold: float = 0.75,
    novelty_threshold: float = 0.2,
    rerank: str = None,
    rerank_prune: bool = True,
    kg_ppr: bool = False,
    aggregate_weights: dict = None,
):
    """Keyword + semantic + temporal search.

    Combines the local trie index with the vector store:
      - ``keywords`` (list of strings, ANDed by default) pre-filters via the
        trie — exact token match, case-insensitive.
      - ``since`` / ``until`` (ISO dates) bound the drawer's ``filed_at``.
      - ``as_of`` (ISO date) returns only drawers whose validity window
        covers that point in time — same predicate as ``mempalace_kg_query``.
      - ``query`` (optional) then runs a Chroma vector query over the
        surviving candidates. Omit ``query`` to get pure keyword/temporal
        results ordered by ``filed_at`` desc.
      - ``model`` (optional) selects the embedding model's collection.
        Unset = palace default. ``"all"`` = RRF fan-out across every
        enabled model (the trie prefilter runs once then fans out).
      - ``rerank`` (optional: "provence" / "bge") runs a cross-encoder
        reranker over the hits before compression. Call
        ``mempalace_list_rerankers`` to see install status.
    """
    return hybrid_search(
        query or "",
        palace_path=_config.palace_path,
        keywords=list(keywords) if keywords else None,
        keyword_mode=keyword_mode,
        since=since,
        until=until,
        as_of=as_of,
        wing=wing,
        room=room,
        n_results=limit,
        model=model,
        compress=compress,
        token_budget=token_budget,
        dup_threshold=dup_threshold,
        sent_threshold=sent_threshold,
        novelty_threshold=novelty_threshold,
        rerank=rerank,
        rerank_prune=rerank_prune,
        enable_kg_ppr=kg_ppr,
        aggregate_weights=aggregate_weights,
    )


def tool_check_duplicate(content: str, threshold: float = 0.9) -> dict:
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        results = col.query(
            query_texts=[content],
            n_results=5,
            include=["metadatas", "documents", "distances"],
        )
        duplicates = []
        if results["ids"] and results["ids"][0]:
            for i, drawer_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i]
                similarity = round(1 - dist, 3)
                if similarity >= threshold:
                    meta = results["metadatas"][0][i]
                    doc = results["documents"][0][i]
                    duplicates.append(
                        {
                            "id": drawer_id,
                            "wing": meta.get("wing", "?"),
                            "room": meta.get("room", "?"),
                            "similarity": similarity,
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        }
                    )
        return {
            "is_duplicate": len(duplicates) > 0,
            "matches": duplicates,
        }
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("tool_check_duplicate failed: %s", e)
        return {"error": str(e)}


def tool_get_aaak_spec() -> dict:
    """Return the AAAK dialect specification."""
    return {"aaak_spec": AAAK_SPEC}


def tool_traverse_graph(start_room: str, max_hops: int = 2) -> dict:
    """Walk the palace graph from a room. Find connected ideas across wings."""
    max_hops = max(1, min(max_hops, 10))
    col = _get_collection()
    if not col:
        return _no_palace()
    return traverse(start_room, col=col, max_hops=max_hops)


def tool_find_tunnels(wing_a: str = None, wing_b: str = None) -> dict:
    """Find rooms that bridge two wings — the hallways connecting domains."""
    try:
        wing_a = _sanitize_optional_name(wing_a, "wing_a")
        wing_b = _sanitize_optional_name(wing_b, "wing_b")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    if not col:
        return _no_palace()
    return find_tunnels(wing_a, wing_b, col=col)


def tool_graph_stats() -> dict:
    """Palace graph overview: nodes, tunnels, edges, connectivity."""
    col = _get_collection()
    if not col:
        return _no_palace()
    return graph_stats(col=col)


def tool_create_tunnel(
    source_wing: str,
    source_room: str,
    target_wing: str,
    target_room: str,
    label: str = "",
    source_drawer_id: str = None,
    target_drawer_id: str = None,
):
    """Create an explicit cross-wing tunnel between two palace locations.

    Use when you notice content in one project relates to another project.
    Example: an API design discussion in project_api connects to the
    database schema in project_database.
    """
    try:
        source_wing = sanitize_name(source_wing, "source_wing")
        source_room = sanitize_name(source_room, "source_room")
        target_wing = sanitize_name(target_wing, "target_wing")
        target_room = sanitize_name(target_room, "target_room")
    except ValueError as e:
        return {"error": str(e)}
    return create_tunnel(
        source_wing,
        source_room,
        target_wing,
        target_room,
        label=label,
        source_drawer_id=source_drawer_id,
        target_drawer_id=target_drawer_id,
    )


def tool_list_tunnels(wing: str = None):
    """List all explicit cross-wing tunnels, optionally filtered by wing."""
    try:
        wing = _sanitize_optional_name(wing, "wing")
    except ValueError as e:
        return {"error": str(e)}
    return list_tunnels(wing)


def tool_delete_tunnel(tunnel_id: str):
    """Delete an explicit tunnel by its ID."""
    if not tunnel_id or not isinstance(tunnel_id, str):
        return {"error": "tunnel_id is required"}
    return delete_tunnel(tunnel_id)


def tool_follow_tunnels(wing: str, room: str):
    """Follow explicit tunnels from a room to see connected drawers in other wings."""
    try:
        wing = sanitize_name(wing, "wing")
        room = sanitize_name(room, "room")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    return follow_tunnels(wing, room, col=col)


# ==================== WRITE TOOLS ====================


def tool_add_drawer(
    wing: str,
    room: str,
    content: str,
    source_file: str = None,
    added_by: str = "mcp",
    model: str = None,
):
    """File verbatim content into a wing/room. Checks for duplicates first.

    ``model`` (optional) selects which embedding model's collection to
    write to. Unset = palace default. ``"all"`` is rejected — writes
    need a concrete destination.
    """
    if model == "all":
        return {"success": False, "error": "model='all' is read-only; pick a concrete slug."}
    col = _get_collection(create=True, model=model)
    if not col:
        return _no_palace()

    drawer_id = (
        f"drawer_{wing}_{room}_"
        f"{hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]}"
    )

    _wal_log(
        "add_drawer",
        {
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
            "added_by": added_by,
            "content_length": len(content),
            "content_preview": content[:200],
        },
    )

    # Idempotency: if the deterministic ID already exists, return success as a no-op.
    try:
        existing = col.get(ids=[drawer_id])
        if existing and existing["ids"]:
            return {"success": True, "reason": "already_exists", "drawer_id": drawer_id}
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("tool_add_drawer: idempotency probe failed — %s", e)

    try:
        from .aggregates import hydrate_drawer_metadata, mark_container_dirty

        metadata = {
            "wing": wing,
            "room": room,
            "source_file": source_file or "",
            "chunk_index": 0,
            "added_by": added_by,
            "filed_at": datetime.now(UTC).isoformat(),
        }
        hydrate_drawer_metadata(metadata, content)
        col.upsert(
            ids=[drawer_id],
            documents=[content],
            metadatas=[metadata],
        )
        try:
            _get_trie_index().add_drawer(drawer_id, content, metadata)
        except (lmdb.Error, OSError, ValueError) as e:
            logger.warning("Trie index add failed for %s: %s", drawer_id, e)
        mark_container_dirty(
            _config.palace_path,
            wing=metadata.get("wing"),
            hall=metadata.get("hall"),
            room=metadata.get("room"),
        )
        logger.info(f"Filed drawer: {drawer_id} → {wing}/{room}")
        return {"success": True, "drawer_id": drawer_id, "wing": wing, "room": room}
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.exception("tool_add_drawer failed")
        return {"success": False, "error": str(e)}


def tool_delete_drawer(drawer_id: str) -> dict:
    """Delete a single drawer by ID."""
    global _metadata_cache
    col = _get_collection()
    if not col:
        return _no_palace()
    existing = col.get(ids=[drawer_id])
    if not existing["ids"]:
        return {"success": False, "error": f"Drawer not found: {drawer_id}"}

    # Log the deletion with the content being removed for audit trail
    deleted_content = existing.get("documents", [""])[0] if existing.get("documents") else ""
    deleted_meta = existing.get("metadatas", [{}])[0] if existing.get("metadatas") else {}
    _wal_log(
        "delete_drawer",
        {
            "drawer_id": drawer_id,
            "deleted_meta": deleted_meta,
            "content_preview": deleted_content[:200],
        },
    )

    try:
        col.delete(ids=[drawer_id])
        try:
            _get_trie_index().delete_drawer(drawer_id)
        except (lmdb.Error, OSError, ValueError) as e:
            logger.warning("Trie index delete failed for %s: %s", drawer_id, e)
        # The container the deleted drawer belonged to now has stale
        # aggregates — mark it dirty so the next rebuild evicts the
        # drawer's text from the aggregate document.
        from .aggregates import mark_container_dirty

        mark_container_dirty(
            _config.palace_path,
            wing=deleted_meta.get("wing"),
            hall=deleted_meta.get("hall"),
            room=deleted_meta.get("room"),
        )
        logger.info(f"Deleted drawer: {drawer_id}")
        return {"success": True, "drawer_id": drawer_id}
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.exception("tool_delete_drawer failed")
        return {"success": False, "error": str(e)}


def tool_get_drawer(drawer_id: str):
    """Fetch a single drawer by ID. Returns full content and metadata."""
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        result = col.get(ids=[drawer_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return {"error": f"Drawer not found: {drawer_id}"}
        meta = result["metadatas"][0]
        doc = result["documents"][0]
        return {
            "drawer_id": drawer_id,
            "content": doc,
            "wing": meta.get("wing", ""),
            "room": meta.get("room", ""),
            "metadata": meta,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_list_drawers(wing: str = None, room: str = None, limit: int = 20, offset: int = 0):
    """List drawers with pagination. Optional wing/room filter."""
    limit = max(1, min(limit, _MAX_RESULTS))
    offset = max(0, offset)
    try:
        wing = _sanitize_optional_name(wing, "wing")
        room = _sanitize_optional_name(room, "room")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        where = None
        conditions = []
        if wing:
            conditions.append({"wing": wing})
        if room:
            conditions.append({"room": room})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        kwargs = {"include": ["documents", "metadatas"], "limit": limit, "offset": offset}
        if where:
            kwargs["where"] = where
        result = col.get(**kwargs)

        drawers = []
        for i, did in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            doc = result["documents"][i]
            drawers.append(
                {
                    "drawer_id": did,
                    "wing": meta.get("wing", ""),
                    "room": meta.get("room", ""),
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                }
            )
        return {
            "drawers": drawers,
            "count": len(drawers),
            "offset": offset,
            "limit": limit,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_update_drawer(drawer_id: str, content: str = None, wing: str = None, room: str = None):
    """Update an existing drawer's content and/or metadata."""
    global _metadata_cache

    if content is None and wing is None and room is None:
        return {"success": True, "drawer_id": drawer_id, "noop": True}

    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        existing = col.get(ids=[drawer_id], include=["documents", "metadatas"])
        if not existing["ids"]:
            return {"success": False, "error": f"Drawer not found: {drawer_id}"}

        old_meta = existing["metadatas"][0]
        old_doc = existing["documents"][0]

        new_doc = old_doc
        if content is not None:
            try:
                new_doc = sanitize_content(content)
            except ValueError as e:
                return {"success": False, "error": str(e)}

        new_meta = dict(old_meta)
        if wing is not None:
            try:
                new_meta["wing"] = sanitize_name(wing, "wing")
            except ValueError as e:
                return {"success": False, "error": str(e)}
        if room is not None:
            try:
                new_meta["room"] = sanitize_name(room, "room")
            except ValueError as e:
                return {"success": False, "error": str(e)}

        _wal_log(
            "update_drawer",
            {
                "drawer_id": drawer_id,
                "old_wing": old_meta.get("wing", ""),
                "old_room": old_meta.get("room", ""),
                "new_wing": new_meta.get("wing", ""),
                "new_room": new_meta.get("room", ""),
                "content_changed": content is not None,
                "content_preview": new_doc[:200] if content is not None else None,
            },
        )

        update_kwargs = {"ids": [drawer_id]}
        if content is not None:
            update_kwargs["documents"] = [new_doc]
        update_kwargs["metadatas"] = [new_meta]
        col.update(**update_kwargs)

        _metadata_cache = None

        logger.info(f"Updated drawer: {drawer_id}")
        return {
            "success": True,
            "drawer_id": drawer_id,
            "wing": new_meta.get("wing", ""),
            "room": new_meta.get("room", ""),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== KNOWLEDGE GRAPH ====================


def tool_kg_query(entity: str, as_of: str = None, direction: str = "both") -> dict:
    """Query the knowledge graph for an entity's relationships."""
    try:
        entity = sanitize_name(entity, "entity")
    except ValueError as e:
        return {"error": str(e)}
    if direction not in ("outgoing", "incoming", "both"):
        return {"error": "direction must be 'outgoing', 'incoming', or 'both'"}
    results = _kg.query_entity(entity, as_of=as_of, direction=direction)
    return {"entity": entity, "as_of": as_of, "facts": results, "count": len(results)}


def tool_kg_add(
    subject: str, predicate: str, object: str, valid_from: str = None, source_closet: str = None
):
    """Add a relationship to the knowledge graph."""
    try:
        subject = sanitize_name(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        object = sanitize_name(object, "object")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    _wal_log(
        "kg_add",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "valid_from": valid_from,
            "source_closet": source_closet,
        },
    )
    triple_id = _kg.add_triple(
        subject, predicate, object, valid_from=valid_from, source_closet=source_closet
    )
    return {"success": True, "triple_id": triple_id, "fact": f"{subject} → {predicate} → {object}"}


def tool_kg_invalidate(subject: str, predicate: str, object: str, ended: str = None) -> dict:
    """Mark a fact as no longer true (set end date)."""
    try:
        subject = sanitize_name(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        object = sanitize_name(object, "object")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    _wal_log(
        "kg_invalidate",
        {"subject": subject, "predicate": predicate, "object": object, "ended": ended},
    )
    _kg.invalidate(subject, predicate, object, ended=ended)
    return {
        "success": True,
        "fact": f"{subject} → {predicate} → {object}",
        "ended": ended or "today",
    }


def tool_kg_timeline(entity: str = None) -> dict:
    """Get chronological timeline of facts, optionally for one entity."""
    if entity is not None:
        try:
            entity = sanitize_name(entity, "entity")
        except ValueError as e:
            return {"error": str(e)}
    results = _kg.timeline(entity)
    return {"entity": entity or "all", "timeline": results, "count": len(results)}


def tool_kg_stats() -> dict:
    """Knowledge graph overview: entities, triples, relationship types."""
    return _kg.stats()


# ==================== AGGREGATES ====================


def tool_aggregates_rebuild(
    level: str = None,
    container: str = None,
    all_containers: bool = False,
    model: str = None,
) -> dict:
    """Rebuild wing/hall/room aggregate embeddings.

    With no args, rebuilds every currently-dirty container. Pass
    ``level`` + ``container`` to rebuild one specific aggregate, or
    ``all_containers=True`` to rebuild the entire palace from scratch
    (slower; useful after bulk ingests). ``model`` optionally scopes
    the rebuild to one embedding-model slug; omit to rebuild every
    enabled slug.
    """
    from . import aggregates as _agg

    slugs = [model] if model else None
    if level and container:
        if level not in _agg.LEVELS:
            return {
                "error": f"level must be one of {_agg.LEVELS}",
                "hint": "retry with level=wing/hall/room and a container name",
            }
        effective_slugs = slugs or (_config.enabled_embedding_models or ["default"])
        written = 0
        for s in effective_slugs:
            written += _agg.rebuild_containers(
                _config.palace_path, level=level, containers=[container], slug=s
            )
        _agg.clear_dirty(_config.palace_path, level, [container])
        return {
            "rebuilt": written,
            "by_level": {level: written},
            "slugs": effective_slugs,
        }

    if all_containers:
        result = _agg.rebuild_all(_config.palace_path, slugs=slugs)
    else:
        result = _agg.rebuild_dirty(_config.palace_path, slugs=slugs)
    return {
        "rebuilt": result["total"],
        "by_level": result["by_level"],
        "slugs": result["slugs"],
    }


def tool_aggregates_status() -> dict:
    """Return dirty-container counts and the last rebuild timestamp."""
    from . import aggregates as _agg

    dirty = _agg.list_dirty(_config.palace_path)
    return {
        "enabled": _config.aggregate_enabled,
        "top_k": _config.aggregate_top_k,
        "weights": _config.aggregate_weights,
        "dirty": {level: len(dirty.get(level, [])) for level in _agg.LEVELS},
        "dirty_containers": dirty,
        "last_rebuild": _agg.latest_rebuilt_any(_config.palace_path),
    }


# ==================== AGENT DIARY ====================


def tool_diary_write(agent_name: str, entry: str, topic: str = "general") -> dict:
    """
    Write a diary entry for this agent. Each agent gets its own wing
    with a diary room. Entries are timestamped and accumulate over time.

    This is the agent's personal journal — observations, thoughts,
    what it worked on, what it noticed, what it thinks matters.
    """
    try:
        agent_name = sanitize_name(agent_name, "agent_name")
        entry = sanitize_content(entry)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    wing = f"wing_{agent_name.lower().replace(' ', '_')}"
    room = "diary"
    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    now = datetime.now(UTC)
    entry_hash = hashlib.md5(entry[:50].encode(), usedforsecurity=False).hexdigest()[:8]
    entry_id = f"diary_{wing}_{now.strftime('%Y%m%d_%H%M%S')}_{entry_hash}"

    metadata = {
        "wing": wing,
        "room": room,
        "hall": "hall_diary",
        "topic": topic,
        "type": "diary_entry",
        "agent": agent_name,
        "filed_at": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
    }

    _wal_log(
        "diary_write",
        {
            "agent_name": agent_name,
            "topic": topic,
            "entry_id": entry_id,
            "entry_preview": entry[:200],
        },
    )

    try:
        col.add(
            ids=[entry_id],
            documents=[entry],
            metadatas=[metadata],
        )
        try:
            _get_trie_index().add_drawer(entry_id, entry, metadata)
        except (lmdb.Error, OSError, ValueError) as e:
            logger.warning("Trie index add failed for diary entry %s: %s", entry_id, e)
        # Diary entries are sentinel-skipped inside mark_container_dirty
        # (room="diary" and hall="hall_diary"), but we still register
        # the wing so wing-level aggregates stay live.
        from .aggregates import mark_container_dirty

        mark_container_dirty(
            _config.palace_path,
            wing=metadata.get("wing"),
            hall=metadata.get("hall"),
            room=metadata.get("room"),
        )
        logger.info(f"Diary entry: {entry_id} → {wing}/diary/{topic}")
        return {
            "success": True,
            "entry_id": entry_id,
            "agent": agent_name,
            "topic": topic,
            "timestamp": now.isoformat(),
        }
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.exception("tool_diary_write failed")
        return {"success": False, "error": str(e)}


def tool_diary_read(agent_name: str, last_n: int = 10) -> dict:
    """
    Read an agent's recent diary entries. Returns the last N entries
    in chronological order — the agent's personal journal.
    """
    try:
        agent_name = sanitize_name(agent_name, "agent_name")
    except ValueError as e:
        return {"error": str(e)}
    last_n = max(1, min(last_n, 100))
    wing = f"wing_{agent_name.lower().replace(' ', '_')}"
    col = _get_collection()
    if not col:
        return _no_palace()

    try:
        results = col.get(
            where={"$and": [{"wing": wing}, {"room": "diary"}]},
            include=["documents", "metadatas"],
            limit=10000,
        )

        if not results["ids"]:
            return {"agent": agent_name, "entries": [], "message": "No diary entries yet."}

        # Combine and sort by timestamp
        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"], strict=False):
            entries.append(
                {
                    "date": meta.get("date", ""),
                    "timestamp": meta.get("filed_at", ""),
                    "topic": meta.get("topic", ""),
                    "content": doc,
                }
            )

        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        entries = entries[:last_n]

        return {
            "agent": agent_name,
            "entries": entries,
            "total": len(results["ids"]),
            "showing": len(entries),
        }
    except (chromadb.errors.ChromaError, ValueError, KeyError) as e:
        logger.debug("tool_diary_read failed: %s", e)
        return {"error": str(e)}


def tool_hook_settings(silent_save: bool | None = None, desktop_toast: bool | None = None) -> dict:
    """Get or set the user's hook behavior knobs.

    Reads/writes ``~/.mempalace/config.json`` via ``MempalaceConfig``. Call
    with no args to view current values; pass ``silent_save`` or
    ``desktop_toast`` to update.
    """
    if silent_save is not None:
        _config.set_hook_setting("silent_save", bool(silent_save))
    if desktop_toast is not None:
        _config.set_hook_setting("desktop_toast", bool(desktop_toast))
    return {
        "silent_save": _config.hook_silent_save,
        "desktop_toast": _config.hook_desktop_toast,
    }


# ==================== MCP PROTOCOL ====================

TOOLS = {
    "mempalace_status": {
        "description": "Palace overview — total drawers, wing and room counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_status,
    },
    "mempalace_list_wings": {
        "description": "List all wings with drawer counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_wings,
    },
    "mempalace_list_rooms": {
        "description": "List rooms within a wing (or all rooms if no wing given)",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing to list rooms for (optional)"},
            },
        },
        "handler": tool_list_rooms,
    },
    "mempalace_get_taxonomy": {
        "description": "Full taxonomy: wing → room → drawer count",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_taxonomy,
    },
    "mempalace_get_aaak_spec": {
        "description": "Get the AAAK dialect specification — the compressed memory format MemPalace uses. Call this if you need to read or write AAAK-compressed memories.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_aaak_spec,
    },
    "mempalace_kg_query": {
        "description": "Query the knowledge graph for an entity's relationships. Returns typed facts with temporal validity. E.g. 'Max' → child_of Alice, loves chess, does swimming. Filter by date with as_of to see what was true at a point in time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to query (e.g. 'Max', 'MyProject', 'Alice')",
                },
                "as_of": {
                    "type": "string",
                    "description": "Date filter — only facts valid at this date (YYYY-MM-DD, optional)",
                },
                "direction": {
                    "type": "string",
                    "description": "outgoing (entity→?), incoming (?→entity), or both (default: both)",
                },
            },
            "required": ["entity"],
        },
        "handler": tool_kg_query,
    },
    "mempalace_kg_add": {
        "description": "Add a fact to the knowledge graph. Subject → predicate → object with optional time window. E.g. ('Max', 'started_school', 'Year 7', valid_from='2026-09-01').",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "The entity doing/being something"},
                "predicate": {
                    "type": "string",
                    "description": "The relationship type (e.g. 'loves', 'works_on', 'daughter_of')",
                },
                "object": {"type": "string", "description": "The entity being connected to"},
                "valid_from": {
                    "type": "string",
                    "description": "When this became true (YYYY-MM-DD, optional)",
                },
                "source_closet": {
                    "type": "string",
                    "description": "Closet ID where this fact appears (optional)",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
        "handler": tool_kg_add,
    },
    "mempalace_kg_invalidate": {
        "description": "Mark a fact as no longer true. E.g. ankle injury resolved, job ended, moved house.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Entity"},
                "predicate": {"type": "string", "description": "Relationship"},
                "object": {"type": "string", "description": "Connected entity"},
                "ended": {
                    "type": "string",
                    "description": "When it stopped being true (YYYY-MM-DD, default: today)",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
        "handler": tool_kg_invalidate,
    },
    "mempalace_kg_timeline": {
        "description": "Chronological timeline of facts. Shows the story of an entity (or everything) in order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to get timeline for (optional — omit for full timeline)",
                },
            },
        },
        "handler": tool_kg_timeline,
    },
    "mempalace_kg_stats": {
        "description": "Knowledge graph overview: entities, triples, current vs expired facts, relationship types.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_kg_stats,
    },
    "mempalace_traverse": {
        "description": "Walk the palace graph from a room. Shows connected ideas across wings — the tunnels. Like following a thread through the palace: start at 'chromadb-setup' in wing_code, discover it connects to wing_myproject (planning) and wing_user (feelings about it).",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_room": {
                    "type": "string",
                    "description": "Room to start from (e.g. 'chromadb-setup', 'riley-school')",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "How many connections to follow (default: 2)",
                },
            },
            "required": ["start_room"],
        },
        "handler": tool_traverse_graph,
    },
    "mempalace_find_tunnels": {
        "description": "Find rooms that bridge two wings — the hallways connecting different domains. E.g. what topics connect wing_code to wing_team?",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing_a": {"type": "string", "description": "First wing (optional)"},
                "wing_b": {"type": "string", "description": "Second wing (optional)"},
            },
        },
        "handler": tool_find_tunnels,
    },
    "mempalace_graph_stats": {
        "description": "Palace graph overview: total rooms, tunnel connections, edges between wings.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_graph_stats,
    },
    "mempalace_create_tunnel": {
        "description": "Create a cross-wing tunnel linking two palace locations. Use when content in one project relates to another — e.g., an API design in project_api connects to a database schema in project_database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_wing": {"type": "string", "description": "Wing of the source"},
                "source_room": {"type": "string", "description": "Room in the source wing"},
                "target_wing": {"type": "string", "description": "Wing of the target"},
                "target_room": {"type": "string", "description": "Room in the target wing"},
                "label": {"type": "string", "description": "Description of the connection"},
                "source_drawer_id": {
                    "type": "string",
                    "description": "Optional specific drawer ID",
                },
                "target_drawer_id": {
                    "type": "string",
                    "description": "Optional specific drawer ID",
                },
            },
            "required": ["source_wing", "source_room", "target_wing", "target_room"],
        },
        "handler": tool_create_tunnel,
    },
    "mempalace_list_tunnels": {
        "description": "List all explicit cross-wing tunnels. Optionally filter by wing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {
                    "type": "string",
                    "description": "Filter tunnels by wing (shows tunnels where wing is source or target)",
                },
            },
        },
        "handler": tool_list_tunnels,
    },
    "mempalace_delete_tunnel": {
        "description": "Delete an explicit tunnel by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tunnel_id": {"type": "string", "description": "Tunnel ID to delete"},
            },
            "required": ["tunnel_id"],
        },
        "handler": tool_delete_tunnel,
    },
    "mempalace_follow_tunnels": {
        "description": "Follow tunnels from a room to see what it connects to in other wings. Returns connected rooms with drawer previews.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing to start from"},
                "room": {"type": "string", "description": "Room to follow tunnels from"},
            },
            "required": ["wing", "room"],
        },
        "handler": tool_follow_tunnels,
    },
    "mempalace_search": {
        "description": "Semantic search. Returns verbatim drawer content with similarity scores. IMPORTANT: 'query' must contain ONLY search keywords. Use 'context' for background. Results with cosine distance > max_distance are filtered out.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search query ONLY — keywords or a question. Max 250 chars.",
                    "maxLength": 250,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                    "minimum": 1,
                    "maximum": 100,
                },
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
                "model": {
                    "type": "string",
                    "description": (
                        "Embedding model slug (optional). Unset = palace default. "
                        "Pick by workload: 'jina-code-v2' for source code "
                        "(CodeSearchNet, 8k context), 'nomic-text-v1.5' for "
                        "long LLM conversations (8k context), 'mxbai-large' "
                        "for MTEB-proven general retrieval, 'bge-small-en' "
                        "for budget general retrieval. Call mempalace_list_models "
                        "to see install/enable status + per-model descriptions. "
                        "'all' = RRF fan-out across every enabled model with "
                        "automatic drawer deduplication (reads only)."
                    ),
                },
                "compress": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "none",
                        "dedupe",
                        "sentences",
                        "aggressive",
                        "llmlingua2",
                    ],
                    "description": (
                        "Result compression mode: auto (default; dedupe on fan-out, "
                        "none otherwise), none, dedupe, sentences, aggressive, or "
                        "llmlingua2 (Microsoft's learned token-level compressor — "
                        "requires the compress-llmlingua optional extra)."
                    ),
                },
                "token_budget": {
                    "type": "integer",
                    "description": (
                        "Max output tokens (aggressive mode only). Halts ingestion "
                        "when cumulative output exceeds this value."
                    ),
                },
                "dup_threshold": {
                    "type": "number",
                    "description": "Drawer-level Jaccard cutoff for dedupe mode (0..1, default 0.7)",
                },
                "sent_threshold": {
                    "type": "number",
                    "description": "Sentence-level bigram Jaccard cutoff for sentences mode (0..1, default 0.75)",
                },
                "novelty_threshold": {
                    "type": "number",
                    "description": "Minimum novel-trigram fraction for aggressive mode (0..1, default 0.2)",
                },
                "rerank": {
                    "type": "string",
                    "enum": ["none", "provence", "bge"],
                    "description": (
                        "Cross-encoder reranker to apply after the vector "
                        "query. 'provence' (unified rerank + per-token "
                        "prune) requires the 'rerank-provence' extra. 'bge' "
                        "(BGE-reranker-v2-m3 via fastembed, no torch) "
                        "requires the 'rerank-bge' extra. Call "
                        "mempalace_list_rerankers for install status."
                    ),
                },
                "rerank_prune": {
                    "type": "boolean",
                    "description": (
                        "When rerank='provence', write pruned_text into "
                        "each hit (default true). Ignored for other "
                        "rerankers or when rerank is unset."
                    ),
                },
                "kg_ppr": {
                    "type": "boolean",
                    "description": (
                        "Enable HippoRAG-style Personalized PageRank "
                        "fusion over the knowledge graph. Extracts "
                        "proper nouns from the query, runs PPR seeded "
                        "on those entities, and unions the top-ranked "
                        "drawers into the vector candidate set. "
                        "Requires the KG to be populated via "
                        "mempalace mine --extract-kg or mempalace "
                        "kg-extract."
                    ),
                },
                "aggregate_weights": {
                    "type": "object",
                    "description": (
                        "Per-level RRF weight multipliers for "
                        "hierarchical aggregate retrieval. Keys: "
                        "'drawer' (base direct-hit weight, default 1.0), "
                        "'room' / 'hall' / 'wing' (container-aggregate "
                        "weights, defaults 0.3/0.2/0.1). Omit to use "
                        "palace defaults; pass {'drawer': 1.0, 'wing': 0, "
                        "'hall': 0, 'room': 0} to disable aggregates for "
                        "one query without touching config."
                    ),
                },
            },
            "required": ["query"],
        },
        "handler": tool_search,
    },
    "mempalace_aggregates_rebuild": {
        "description": (
            "Recompute wing/hall/room aggregate embeddings. With no "
            "args, rebuilds every currently-dirty container. Pass "
            "`level` + `container` to rebuild one specific aggregate, "
            "or `all_containers=true` to rebuild everything from "
            "scratch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["wing", "hall", "room"],
                    "description": "Which taxonomic level to rebuild (requires container)",
                },
                "container": {
                    "type": "string",
                    "description": "Container name to rebuild (e.g. 'wing_auth', 'hall_technical')",
                },
                "all_containers": {
                    "type": "boolean",
                    "description": "Rebuild every container in the palace (slower)",
                },
                "model": {
                    "type": "string",
                    "description": "Scope to one embedding-model slug (default: all enabled)",
                },
            },
        },
        "handler": tool_aggregates_rebuild,
    },
    "mempalace_aggregates_status": {
        "description": (
            "Report dirty/clean aggregate counts per level, the last "
            "rebuild timestamp, and the currently-configured "
            "aggregate weights."
        ),
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_aggregates_status,
    },
    "mempalace_hybrid_search": {
        "description": (
            "Hybrid search over the palace: keyword (local trie) + semantic "
            "(ChromaDB) + temporal. Keywords are ANDed by default. "
            "`since`/`until` bound the drawer's filed_at; `as_of` returns "
            "drawers whose validity window covers that date. Omit `query` "
            "to get pure keyword/temporal results ordered by filed_at desc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic query (optional — omit for pure keyword/temporal)",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords to require (exact tokens, case-insensitive)",
                },
                "keyword_mode": {
                    "type": "string",
                    "description": "all (AND, default), any (OR), or prefix (AND on prefix match)",
                },
                "since": {
                    "type": "string",
                    "description": "Only drawers filed on or after this date (ISO)",
                },
                "until": {
                    "type": "string",
                    "description": "Only drawers filed on or before this date (ISO)",
                },
                "as_of": {
                    "type": "string",
                    "description": "Point-in-time — validity window must cover this date",
                },
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
                "limit": {"type": "integer", "description": "Max results (default 5)"},
                "model": {
                    "type": "string",
                    "description": (
                        "Embedding model slug (optional). Unset = palace default. "
                        "Pick by workload: 'jina-code-v2' for source code "
                        "(CodeSearchNet, 8k context), 'nomic-text-v1.5' for "
                        "long LLM conversations (8k context), 'mxbai-large' "
                        "for MTEB-proven general retrieval, 'bge-small-en' "
                        "for budget general retrieval. Call mempalace_list_models "
                        "to see install/enable status + per-model descriptions. "
                        "'all' = RRF fan-out across every enabled model with "
                        "automatic drawer deduplication (reads only)."
                    ),
                },
                "compress": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "none",
                        "dedupe",
                        "sentences",
                        "aggressive",
                        "llmlingua2",
                    ],
                    "description": (
                        "Result compression mode: auto (default; dedupe on fan-out, "
                        "none otherwise), none, dedupe, sentences, aggressive, or "
                        "llmlingua2 (Microsoft's learned token-level compressor — "
                        "requires the compress-llmlingua optional extra)."
                    ),
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Max output tokens (aggressive mode only).",
                },
                "dup_threshold": {
                    "type": "number",
                    "description": "Drawer-level Jaccard cutoff for dedupe mode (0..1, default 0.7)",
                },
                "sent_threshold": {
                    "type": "number",
                    "description": "Sentence-level bigram Jaccard cutoff (0..1, default 0.75)",
                },
                "novelty_threshold": {
                    "type": "number",
                    "description": "Minimum novel-trigram fraction for aggressive mode (0..1, default 0.2)",
                },
                "rerank": {
                    "type": "string",
                    "enum": ["none", "provence", "bge"],
                    "description": (
                        "Cross-encoder reranker to apply after the vector "
                        "query. 'provence' unified rerank + pruning or "
                        "'bge' pure rerank. Call mempalace_list_rerankers "
                        "for install status."
                    ),
                },
                "rerank_prune": {
                    "type": "boolean",
                    "description": (
                        "When rerank='provence', enable per-token pruning (default true)."
                    ),
                },
                "kg_ppr": {
                    "type": "boolean",
                    "description": (
                        "Enable HippoRAG PPR fusion over the knowledge "
                        "graph. See mempalace_search for full details."
                    ),
                },
                "aggregate_weights": {
                    "type": "object",
                    "description": (
                        "Per-level RRF weight multipliers for "
                        "hierarchical aggregate retrieval. See "
                        "mempalace_search for details."
                    ),
                },
            },
        },
        "handler": tool_hybrid_search,
    },
    "mempalace_check_duplicate": {
        "description": "Check if content already exists in the palace before filing",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to check"},
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold 0-1 (default 0.9)",
                },
            },
            "required": ["content"],
        },
        "handler": tool_check_duplicate,
    },
    "mempalace_add_drawer": {
        "description": "File verbatim content into the palace. Checks for duplicates first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing (project name)"},
                "room": {
                    "type": "string",
                    "description": "Room (aspect: backend, decisions, meetings...)",
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim content to store — exact words, never summarized",
                },
                "source_file": {"type": "string", "description": "Where this came from (optional)"},
                "added_by": {"type": "string", "description": "Who is filing this (default: mcp)"},
                "model": {
                    "type": "string",
                    "description": (
                        "Embedding model slug (optional). Unset = palace default. "
                        "Pick by workload: 'jina-code-v2' for source code, "
                        "'nomic-text-v1.5' for long LLM conversations, "
                        "'mxbai-large' for MTEB-proven retrieval. Call "
                        "mempalace_list_models for per-model descriptions. "
                        "'all' is rejected on writes — pick a concrete slug."
                    ),
                },
            },
            "required": ["wing", "room", "content"],
        },
        "handler": tool_add_drawer,
    },
    "mempalace_list_models": {
        "description": (
            "List every embedding model in the registry with its install "
            "status (optional extras present), enable status (in config), "
            "default flag, drawer count per collection, and a description "
            "field you can use to pick the right model for the user's "
            "workload (e.g. 'jina-code-v2' for code repos, 'nomic-text-v1.5' "
            "for long LLM conversations, 'mxbai-large' for MTEB-proven "
            "general retrieval). Call this before suggesting a specific "
            "--model or before recommending `mempalace models enable <slug>`."
        ),
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_models,
    },
    "mempalace_list_rerankers": {
        "description": (
            "List every cross-encoder reranker registered in MemPalace "
            "with its install status and pruning support. Two rerankers "
            "ship: 'provence' (unified rerank + per-token context prune "
            "via DeBERTa-v3, ICLR 2025 — requires the 'rerank-provence' "
            "extra) and 'bge' (BAAI/bge-reranker-v2-m3 via fastembed "
            "ONNX, no torch — requires the 'rerank-bge' extra). Call "
            "this before suggesting rerank=<slug> on mempalace_search or "
            "mempalace_hybrid_search. Each entry carries a description "
            "field so you can match the reranker to the workload."
        ),
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_rerankers,
    },
    "mempalace_delete_drawer": {
        "description": "Delete a drawer by ID. Irreversible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to delete"},
            },
            "required": ["drawer_id"],
        },
        "handler": tool_delete_drawer,
    },
    "mempalace_get_drawer": {
        "description": "Fetch a single drawer by ID — returns full content and metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to fetch"},
            },
            "required": ["drawer_id"],
        },
        "handler": tool_get_drawer,
    },
    "mempalace_list_drawers": {
        "description": "List drawers with pagination. Optional wing/room filter. Returns IDs, wings, rooms, and content previews.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
                "limit": {
                    "type": "integer",
                    "description": "Max results per page (default 20, max 100)",
                    "minimum": 1,
                    "maximum": 100,
                },
                "offset": {
                    "type": "integer",
                    "description": "Offset for pagination (default 0)",
                    "minimum": 0,
                },
            },
        },
        "handler": tool_list_drawers,
    },
    "mempalace_update_drawer": {
        "description": "Update an existing drawer's content and/or metadata (wing, room). Fetches existing drawer first; returns error if not found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to update"},
                "content": {
                    "type": "string",
                    "description": "New content (optional — omit to keep existing)",
                },
                "wing": {
                    "type": "string",
                    "description": "New wing (optional — omit to keep existing)",
                },
                "room": {
                    "type": "string",
                    "description": "New room (optional — omit to keep existing)",
                },
            },
            "required": ["drawer_id"],
        },
        "handler": tool_update_drawer,
    },
    "mempalace_diary_write": {
        "description": "Write to your personal agent diary in AAAK format. Your observations, thoughts, what you worked on, what matters. Each agent has their own diary with full history. Write in AAAK for compression — e.g. 'SESSION:2026-04-04|built.palace.graph+diary.tools|ALC.req:agent.diaries.in.aaak|★★★'. Use entity codes from the AAAK spec.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary wing",
                },
                "entry": {
                    "type": "string",
                    "description": "Your diary entry in AAAK format — compressed, entity-coded, emotion-marked",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic tag (optional, default: general)",
                },
            },
            "required": ["agent_name", "entry"],
        },
        "handler": tool_diary_write,
    },
    "mempalace_diary_read": {
        "description": "Read your recent diary entries (in AAAK). See what past versions of yourself recorded — your journal across sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary wing",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent entries to read (default: 10)",
                },
            },
            "required": ["agent_name"],
        },
        "handler": tool_diary_read,
    },
    "mempalace_hook_settings": {
        "description": (
            "Get or set hook behavior. silent_save: True = save directly "
            "(no MCP clutter), False = legacy blocking. desktop_toast: "
            "True = show desktop notification. Call with no args to view."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "silent_save": {
                    "type": "boolean",
                    "description": "True = silent direct save, False = blocking MCP calls",
                },
                "desktop_toast": {
                    "type": "boolean",
                    "description": "True = show desktop toast via notify-send",
                },
            },
        },
        "handler": tool_hook_settings,
    },
}


SUPPORTED_PROTOCOL_VERSIONS = [
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
]


def handle_request(request):
    method = request.get("method") or ""
    params = request.get("params") or {}
    req_id = request.get("id")

    if method == "initialize":
        client_version = params.get("protocolVersion", SUPPORTED_PROTOCOL_VERSIONS[-1])
        negotiated = (
            client_version
            if client_version in SUPPORTED_PROTOCOL_VERSIONS
            else SUPPORTED_PROTOCOL_VERSIONS[0]
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace", "version": __version__},
            },
        }
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    elif method.startswith("notifications/"):
        # Notifications (no id) never get a response per JSON-RPC spec
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": n, "description": t["description"], "inputSchema": t["input_schema"]}
                    for n, t in TOOLS.items()
                ]
            },
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        # Whitelist arguments to declared schema properties only.
        # Prevents callers from spoofing internal params like added_by/source_file.
        # Skip filtering if handler explicitly accepts **kwargs (pass-through).
        # Default to filtering on inspect failure (safe fallback).
        import inspect

        schema_props = TOOLS[tool_name]["input_schema"].get("properties", {})
        try:
            handler = TOOLS[tool_name]["handler"]
            sig = inspect.signature(handler)
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            accepts_var_keyword = False
        if not accepts_var_keyword:
            tool_args = {k: v for k, v in tool_args.items() if k in schema_props}
        # Coerce argument types based on input_schema.
        # MCP JSON transport may deliver integers as floats or strings;
        # ChromaDB and Python slicing require native int.
        for key, value in list(tool_args.items()):
            prop_schema = schema_props.get(key, {})
            declared_type = prop_schema.get("type")
            try:
                if declared_type == "integer" and not isinstance(value, int):
                    tool_args[key] = int(value)
                elif declared_type == "number" and not isinstance(value, (int, float)):
                    tool_args[key] = float(value)
            except (ValueError, TypeError):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Invalid value for parameter '{key}'"},
                }
        try:
            tool_args.pop("wait_for_previous", None)
            result = TOOLS[tool_name]["handler"](**tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception:
            # Broad catch: this is the MCP protocol boundary. Any exception
            # from any handler must be translated to a JSON-RPC error so the
            # server doesn't crash mid-session. Full traceback is logged.
            logger.exception(f"Tool error in {tool_name}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": "Internal tool error"},
            }

    # Notifications (missing id) must never get a response
    if req_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main():  # pragma: no cover
    logger.info("MemPalace MCP Server starting...")

    # Warm the trie bitmap LRU on startup so the first query for any hot
    # token hits the in-process cache instead of paying the ~5–10 μs
    # LMDB read + deserialize cost. The MCP server is long-lived so the
    # one-time ~20–50 ms warmup is amortized over thousands of queries.
    # Non-fatal: if the palace or trie doesn't exist yet, skip silently.
    try:
        trie = _get_trie_index()
        if trie is not None:
            loaded = trie.warm()
            if loaded:
                logger.info(f"Trie warm: loaded {loaded} hot bitmaps")
    except (lmdb.Error, OSError, ValueError) as e:
        logger.debug(f"Trie warm skipped: {e}")

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except json.JSONDecodeError as e:
            logger.error(f"MCP protocol error: malformed JSON on stdin: {e}")
        except (OSError, BrokenPipeError) as e:
            logger.error(f"MCP I/O error: {e}")
            break
        except Exception as e:
            # Broad catch: the main stdin loop must never crash — keep the
            # server alive so the MCP client can retry. Any uncaught error
            # from handle_request bubbles up here and is surfaced as a log
            # line.
            logger.exception(f"Server error: {e}")


if __name__ == "__main__":
    main()
