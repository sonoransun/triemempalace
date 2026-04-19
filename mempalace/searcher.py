#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Semantic search against the palace, with optional keyword + temporal
prefilter via ``trie_index.TrieIndex``. A single entry point
(:func:`hybrid_search`) serves four modes:

* **Pure semantic** — only ``query`` is set; Chroma does the vector search.
* **Pure keyword/temporal** — ``query=""``; trie returns a bitmap sorted
  by ``filed_at`` descending.
* **Hybrid** — ``query`` + keyword/temporal constraints; trie prefilters
  then Chroma ranks within the candidate set.
* **Fan-out** (``model="all"``) — runs the chosen mode against *every*
  enabled model's collection and merges the result lists with
  Reciprocal Rank Fusion.

Returns verbatim text — the actual words, never summaries.
"""

import logging
from pathlib import Path

import chromadb.errors

from .compress import compress_results, resolve_auto_mode
from .palace_io import open_collection
from .trie_index import TrieIndex, trie_db_path

logger = logging.getLogger("mempalace_mcp")


# Reciprocal Rank Fusion constant. 60 is the value from the original
# Cormack/Clarke/Buettcher paper and the Elasticsearch / Vespa default.
_RRF_K = 60


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


def _trie_db_path(palace_path: str) -> str:
    """Colocate the trie next to the palace, matching how the KG does it."""
    return trie_db_path(palace_path)


def _build_where(wing, room, extra_clauses=None):
    """Compose a Chroma ``where`` filter from optional scope clauses."""
    clauses = []
    if wing is not None:
        clauses.append({"wing": wing})
    if room is not None:
        clauses.append({"room": room})
    if extra_clauses:
        clauses.extend(extra_clauses)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _hit_from_meta(doc: str, meta: dict, distance: float | None) -> dict:
    return {
        "text": doc,
        "wing": meta.get("wing", "unknown"),
        "room": meta.get("room", "unknown"),
        "hall": meta.get("hall"),
        "source_file": Path(meta.get("source_file", "?")).name,
        "similarity": round(1 - distance, 3) if distance is not None else None,
        "filed_at": meta.get("filed_at"),
    }


def _filters_block(*, wing, room, keywords, keyword_mode, since, until, as_of, model):
    """Normalized ``filters`` sub-dict for the response envelope."""
    return {
        "wing": wing,
        "room": room,
        "keywords": list(keywords or []),
        "keyword_mode": keyword_mode,
        "temporal": {"since": since, "until": until, "as_of": as_of},
        "model": model,
    }


def _strip_internal_fields(hits: list) -> list:
    """Drop internal-only hit fields (like ``_drawer_id``) before
    serializing the response envelope. Called at the end of every
    search path so the public API shape stays clean.
    """
    out = []
    for h in hits:
        clean = {k: v for k, v in h.items() if not k.startswith("_")}
        out.append(clean)
    return out


def _serialize_ppr_envelope(envelope: dict | None) -> dict:
    """Convert the internal PPR envelope into a JSON-safe response block.

    The internal envelope carries a ``set`` of drawer IDs which is not
    JSON-serializable. This helper converts it to a sorted list and
    caps the list at 100 IDs to keep the response envelope bounded.
    Returns a zero-valued stub when ``envelope`` is ``None`` (PPR
    was never run) so the response shape is consistent.
    """
    if envelope is None:
        return {
            "enabled": False,
            "seeds": [],
            "top_entities": [],
            "drawer_ids": [],
            "drawer_id_count": 0,
            "skipped_reason": None,
        }
    drawer_ids = sorted(envelope.get("drawer_ids") or [])
    return {
        "enabled": True,
        "seeds": list(envelope.get("seeds") or []),
        "top_entities": list(envelope.get("top_entities") or []),
        "drawer_ids": drawer_ids[:100],
        "drawer_id_count": len(drawer_ids),
        "skipped_reason": envelope.get("skipped_reason"),
    }


def hybrid_search(
    query: str,
    palace_path: str,
    *,
    keywords: list | None = None,
    keyword_mode: str = "all",
    since: str | None = None,
    until: str | None = None,
    as_of: str | None = None,
    wing: str | None = None,
    room: str | None = None,
    n_results: int = 5,
    model: str | None = None,
    compress: str = "auto",
    token_budget: int | None = None,
    dup_threshold: float = 0.7,
    sent_threshold: float = 0.75,
    novelty_threshold: float = 0.2,
    rerank: str | None = None,
    rerank_prune: bool = True,
    enable_kg_ppr: bool = False,
    aggregate_weights: dict[str, float] | None = None,
) -> dict:
    """Combine trie-based keyword/temporal filtering with Chroma vector search.

    ``model``:
        * ``None`` → use the palace's configured default embedding model.
        * A specific slug (e.g. ``"jina-code-v2"``) → query that model's
          collection only.
        * ``"all"`` → fan out across every enabled model, merge with
          Reciprocal Rank Fusion.

    ``compress``:
        * ``"auto"`` (default) → ``"dedupe"`` for fan-out, ``"none"``
          otherwise. Backward-compatible default for existing callers.
        * ``"none"`` → passthrough. No compression pass.
        * ``"dedupe"`` → drawer-level clustering by token-set Jaccard.
        * ``"sentences"`` → dedupe + sentence-level shingle dedupe.
        * ``"aggressive"`` → sentences + novelty gate + optional
          ``token_budget`` enforcement.

    ``rerank``:
        * ``None`` (default) → no cross-encoder reranking, behavior
          matches the legacy search path byte-for-byte.
        * ``"provence"`` → unified rerank + per-token pruning via
          ``naver/provence-reranker-debertav3-v1``. Requires the
          ``rerank-provence`` optional extra. When ``rerank_prune=True``
          (default), each hit gains a ``pruned_text`` field that
          downstream compression prefers over ``text``.
        * ``"bge"`` → pure cross-encoder rerank via
          ``BAAI/bge-reranker-v2-m3`` through fastembed ONNX.
          Requires the ``rerank-bge`` optional extra. No pruning;
          the compression stage handles that separately.

    ``rerank_prune`` is ignored when ``rerank="bge"`` or ``rerank=None``.
    """
    resolved_compress = resolve_auto_mode(model=model, compress=compress)

    # When compression is active, overfetch from the underlying search
    # so the clustering stage has enough material to actually collapse
    # redundancy. When compress="none" we stick to ``n_results`` to
    # preserve the exact pre-compression output shape.
    internal_limit = n_results if resolved_compress == "none" else max(n_results * 3, 20)

    if model == "all":
        result = _hybrid_search_fan_out(
            query,
            palace_path,
            keywords=keywords,
            keyword_mode=keyword_mode,
            since=since,
            until=until,
            as_of=as_of,
            wing=wing,
            room=room,
            n_results=internal_limit,
            enable_kg_ppr=enable_kg_ppr,
            aggregate_weights=aggregate_weights,
        )
    else:
        result = _hybrid_search_single(
            query,
            palace_path,
            keywords=keywords,
            keyword_mode=keyword_mode,
            since=since,
            until=until,
            as_of=as_of,
            wing=wing,
            room=room,
            n_results=internal_limit,
            model=model,
            enable_kg_ppr=enable_kg_ppr,
            aggregate_weights=aggregate_weights,
        )

    if "error" in result:
        return result

    # ── Rerank pass (optional) ───────────────────────────────────────
    # Runs after the search but before compression so the compression
    # stage sees the rerank-ordered hits. When ``rerank="provence"``
    # and the reranker supports pruning, each hit gains a
    # ``pruned_text`` field that downstream compression prefers over
    # ``text`` (see compress.py). When ``rerank="bge"`` the hits are
    # reordered by cross-encoder score but text is unchanged.
    raw_hits = result.get("results", [])
    rerank_stats: dict = {"mode": "none", "hits_reranked": 0}
    if rerank is not None and raw_hits:
        from . import rerank as _rerank_module

        try:
            reranker = _rerank_module.load_reranker(rerank)
            raw_hits = reranker.rerank(query, raw_hits, top_k=None, prune=rerank_prune)
            rerank_stats = {
                "mode": rerank,
                "hits_reranked": len(raw_hits),
                "prune": rerank_prune and _rerank_module.get_reranker_spec(rerank).supports_pruning,
            }
            # Provence populates ``pruned_text`` — make the compression
            # stage see the pruned version by aliasing it into ``text``
            # while keeping the original under ``_original_text`` so
            # nothing is lost. ``_original_text`` is stripped by
            # ``_strip_internal_fields`` before the response leaves.
            if rerank_stats.get("prune"):
                for hit in raw_hits:
                    pruned = hit.get("pruned_text")
                    if pruned is not None and pruned != hit.get("text"):
                        hit["_original_text"] = hit.get("text", "")
                        hit["text"] = pruned
        except _rerank_module._MissingExtrasError as e:
            logger.warning("Rerank %r skipped: %s", rerank, e)
            rerank_stats = {
                "mode": "none",
                "hits_reranked": 0,
                "error": str(e),
            }
        except Exception as e:
            # Broad catch: reranking is optional; any backend failure
            # should not break the core search path. Log + degrade.
            logger.warning("Rerank %r failed: %s — returning unranked", rerank, e)
            rerank_stats = {
                "mode": "none",
                "hits_reranked": 0,
                "error": str(e),
            }

    # ── Compression pass ─────────────────────────────────────────────
    # Runs over whatever hit list the (possibly reranked) search
    # produced. For `compress="none"` this is a zero-cost passthrough;
    # for every other mode it clusters, dedupes sentences, and
    # optionally gates by novelty/budget.
    compressed_hits, compression_stats = compress_results(
        raw_hits,
        mode=resolved_compress,
        token_budget=token_budget,
        dup_threshold=dup_threshold,
        sent_threshold=sent_threshold,
        novelty_threshold=novelty_threshold,
    )
    # Truncate to n_results AFTER compression so callers get N
    # maximally-dense hits, not N raw hits that dedupe down to 3.
    compressed_hits = compressed_hits[:n_results]
    # Strip internal fields (e.g. `_drawer_id`, `_original_text`) from
    # the public hits.
    result["results"] = _strip_internal_fields(compressed_hits)
    result["compression"] = compression_stats
    result["rerank"] = rerank_stats
    # Drop top-level internal keys (currently ``_agg_boost`` from the
    # aggregate-fusion pass) before returning to the caller.
    for k in list(result.keys()):
        if k.startswith("_"):
            result.pop(k, None)
    return result


def _hybrid_search_single(
    query: str,
    palace_path: str,
    *,
    keywords: list | None,
    keyword_mode: str,
    since: str | None,
    until: str | None,
    as_of: str | None,
    wing: str | None,
    room: str | None,
    n_results: int,
    model: str | None,
    enable_kg_ppr: bool = False,
    aggregate_weights: dict[str, float] | None = None,
) -> dict:
    """Single-model hybrid search. Called by ``hybrid_search`` and by the
    fan-out path once per enabled model.
    """
    # ── Open the palace ───────────────────────────────────────────────
    try:
        col = open_collection(palace_path, model=model)
    except (OSError, chromadb.errors.ChromaError) as e:
        logger.error("No palace found at %s (model=%s): %s", palace_path, model, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    has_trie_constraint = bool(keywords) or since or until or as_of
    candidate_ids: set | None = None
    kg_ppr_envelope: dict | None = None

    if has_trie_constraint:
        trie = TrieIndex(db_path=_trie_db_path(palace_path))
        candidate_ids = trie.keyword_search(
            list(keywords or []),
            mode=keyword_mode,
            wing=wing,
            room=room,
            since=since,
            until=until,
            as_of=as_of,
        )

    # ── Optional KG-PPR fusion ────────────────────────────────────────
    # When enabled, run Personalized PageRank seeded on query
    # entities and union the resulting drawer IDs into the candidate
    # set. This lets the semantic query rank within {trie ∪ PPR}
    # rather than {trie alone}. See ``mempalace/kg_ppr.py`` and the
    # HippoRAG paper (Gutiérrez et al., NeurIPS 2024).
    if enable_kg_ppr and query and query.strip():
        try:
            from . import kg_ppr as _kg_ppr_module
            from .knowledge_graph import DEFAULT_KG_PATH

            kg_ppr_envelope = _kg_ppr_module.kg_ppr_candidates(query, kg_db_path=DEFAULT_KG_PATH)
            ppr_drawer_ids = kg_ppr_envelope.get("drawer_ids") or set()
            if ppr_drawer_ids:
                if candidate_ids is None:
                    candidate_ids = set(ppr_drawer_ids)
                else:
                    candidate_ids = set(candidate_ids) | set(ppr_drawer_ids)
        except Exception as e:
            # Broad catch: PPR is optional; any failure must degrade
            # gracefully to the normal search path.
            logger.warning("kg_ppr fusion skipped: %s", e)
            kg_ppr_envelope = {
                "seeds": [],
                "top_entities": [],
                "drawer_ids": set(),
                "skipped_reason": f"error: {e}",
            }

    if has_trie_constraint and not candidate_ids:
        return {
            "query": query,
            "filters": _filters_block(
                wing=wing,
                room=room,
                keywords=keywords,
                keyword_mode=keyword_mode,
                since=since,
                until=until,
                as_of=as_of,
                model=model,
            ),
            "results": [],
            "kg_ppr": _serialize_ppr_envelope(kg_ppr_envelope),
        }

    # ── Pure keyword / temporal path ──────────────────────────────────
    has_query = bool(query and query.strip())
    if has_trie_constraint and not has_query:
        trie = TrieIndex(db_path=_trie_db_path(palace_path))
        meta_by_id = trie.get_drawer_meta(candidate_ids)
        ordered = sorted(
            candidate_ids,
            key=lambda did: meta_by_id.get(did, {}).get("filed_at") or "",
            reverse=True,
        )[:n_results]

        docs_by_id: dict[str, tuple[str, dict]] = {}
        if ordered:
            try:
                page = col.get(ids=list(ordered), include=["documents", "metadatas"])
                for did, doc, meta in zip(
                    page["ids"], page["documents"], page["metadatas"], strict=False
                ):
                    docs_by_id[did] = (doc, meta)
            except Exception as e:
                # Broad catch: col may be a test double raising any type.
                logger.error("Chroma fetch for trie hits failed: %s", e)

        hits = []
        for did in ordered:
            doc, meta = docs_by_id.get(did, ("", meta_by_id.get(did, {})))
            hits.append(_hit_from_meta(doc, meta, distance=None))

        return {
            "query": query,
            "filters": _filters_block(
                wing=wing,
                room=room,
                keywords=keywords,
                keyword_mode=keyword_mode,
                since=since,
                until=until,
                as_of=as_of,
                model=model,
            ),
            "results": hits,
            "kg_ppr": _serialize_ppr_envelope(kg_ppr_envelope),
        }

    # ── Semantic path (optionally gated by candidate_ids) ─────────────
    where = _build_where(wing, room)

    # Overfetch budget when the trie has narrowed the candidate set.
    #
    # The old logic was a fixed ``n_results * 4`` multiplier regardless
    # of the trie's selectivity, which caused a recall leak: if the
    # trie returned 80 candidates and the user asked for n_results=5,
    # we'd fetch Chroma's top 20 and then post-filter — leaving most
    # of the 80 candidates unranked. With a filter that selective we
    # should fetch *all* the candidates so Chroma ranks within the
    # whitelist, not the entire collection.
    #
    # New rule:
    #   - If candidate_ids is small (<= candidate_overfetch_cap), fetch
    #     every candidate so Chroma ranks the whole whitelist.
    #   - Otherwise, keep a generous 4× overfetch (up from the old
    #     behavior's exact 4×, now with a floor so small queries still
    #     get headroom).
    candidate_overfetch_cap = 500

    if candidate_ids is not None:
        query_n = max(
            n_results * 4,
            min(len(candidate_ids), candidate_overfetch_cap),
        )
    else:
        query_n = n_results

    # Hybrid retrieval: always query drawers directly (the floor), then use
    # closet hits to boost rankings. Closets are a ranking SIGNAL, never a
    # GATE — direct drawer search is always the baseline.
    #
    # This avoids the "weak-closets regression" where narrative content
    # produces low-signal closets (regex extraction matches few topics)
    # and closet-first routing hides drawers that direct search would find.
    try:
        dkwargs = {
            "query_texts": [query],
            "n_results": query_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            dkwargs["where"] = where
        results = col.query(**dkwargs)
    except Exception as e:
        # Broad catch: the collection may be a real Chroma handle (raising
        # ChromaError) or a test double (raising RuntimeError / anything).
        # Either way we translate to the hybrid search error-dict contract.
        logger.debug("Chroma query failed: %s", e)
        return {"error": f"Search error: {e}"}

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    # ── Aggregate-level boosts ────────────────────────────────────────
    # Compute a per-drawer additive boost from the wing/hall/room
    # aggregate collections and stash it on each hit as ``_agg_boost``.
    # The fan-out RRF merger consumes ``_agg_boost`` directly; the
    # single-model path applies it in the final sort below so the
    # container signal re-ranks drawers before the caller compresses.
    from . import aggregates as _agg_module
    from .config import DEFAULT_AGGREGATE_WEIGHTS as _DEFAULT_AGG_WEIGHTS
    from .config import MempalaceConfig as _Cfg

    cfg_local = _Cfg()
    if aggregate_weights is None:
        effective_agg_weights = getattr(cfg_local, "aggregate_weights", _DEFAULT_AGG_WEIGHTS)
    else:
        effective_agg_weights = aggregate_weights
    resolved_slug = model or getattr(cfg_local, "default_embedding_model", "default")
    agg_boost = _agg_module.aggregate_contributions(
        query,
        palace_path,
        slug=resolved_slug,
        candidate_ids=candidate_ids,
        weights=effective_agg_weights,
    )

    hits: list[dict] = []
    raw_order: list[tuple[str, str, dict, float | None]] = []
    for did, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
        if candidate_ids is not None and did not in candidate_ids:
            continue
        raw_order.append((did, doc, meta, dist))

    # When the aggregate layer is live, re-rank by a composite score so
    # drawers whose wing/hall/room aligns with the query can rise above
    # drawers with a lone strong vector match. When the aggregate layer
    # is disabled (empty boost dict) the composite collapses to the
    # original distance-sorted order, preserving legacy behavior
    # byte-for-byte.
    drawer_weight = float(effective_agg_weights.get("drawer", 1.0))
    if agg_boost:
        # Chroma returns distances in ascending-similar order, so a low
        # distance corresponds to a high similarity. Normalize to a
        # same-direction "score" and fold in the aggregate boost.
        def _composite(entry):
            did, _doc, _meta, dist = entry
            sim = (1.0 - dist) if dist is not None else 0.0
            return drawer_weight * sim + agg_boost.get(did, 0.0)

        raw_order.sort(key=_composite, reverse=True)

    for did, doc, meta, dist in raw_order:
        hit = _hit_from_meta(doc, meta, distance=dist)
        hit["_drawer_id"] = did  # used by fan-out for RRF; stripped on serialize
        if agg_boost.get(did, 0.0) > 0:
            hit["aggregate_boost"] = round(agg_boost[did], 6)
        hits.append(hit)
        if len(hits) >= n_results:
            break

    return {
        "query": query,
        "filters": _filters_block(
            wing=wing,
            room=room,
            keywords=keywords,
            keyword_mode=keyword_mode,
            since=since,
            until=until,
            as_of=as_of,
            model=model,
        ),
        "results": hits,
        "kg_ppr": _serialize_ppr_envelope(kg_ppr_envelope),
        "_agg_boost": agg_boost,
    }


def _hybrid_search_fan_out(
    query: str,
    palace_path: str,
    *,
    keywords: list | None,
    keyword_mode: str,
    since: str | None,
    until: str | None,
    as_of: str | None,
    wing: str | None,
    room: str | None,
    n_results: int,
    enable_kg_ppr: bool = False,
    aggregate_weights: dict[str, float] | None = None,
) -> dict:
    """Cross-model fan-out via Reciprocal Rank Fusion.

    Queries every enabled model's collection, collects the top-K from
    each, and merges with RRF:

        score(drawer) = sum_i 1 / (k + rank_i)

    where k = _RRF_K = 60 and rank_i is the 0-indexed position of the
    drawer in the i-th model's result list. A drawer that appears in
    multiple collections naturally ranks higher because its RRF
    contributions sum.

    The trie prefilter runs inside each per-model call, but because the
    trie itself is embedding-agnostic every call gets the same filter
    result — the cost is constant regardless of how many models are
    enabled.

    Per-model queries run **concurrently** via a thread pool. Chroma's
    Python client is thread-safe for reads, and the embedding adapters
    hold their own internal locks when lazy-loading model weights.
    The fan-out wall-clock drops from roughly
    ``sum(per_model_latency)`` to ``max(per_model_latency)`` — a
    ~N× speedup when N models are enabled.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .config import DEFAULT_AGGREGATE_WEIGHTS, MempalaceConfig

    cfg = MempalaceConfig()
    enabled = cfg.enabled_embedding_models
    if not enabled:
        enabled = ["default"]

    # Defensive ``getattr``: tests monkeypatch MempalaceConfig with a
    # minimal ``_Stub`` that doesn't carry the aggregate-layer knobs,
    # same pattern used for ``fan_out_max_workers`` below.
    if aggregate_weights is None:
        effective_agg_weights = getattr(cfg, "aggregate_weights", DEFAULT_AGGREGATE_WEIGHTS)
    else:
        effective_agg_weights = aggregate_weights

    # Auto-rebuild trigger. When the dirty-container count crosses the
    # configured threshold, fire an async rebuild so the next query
    # sees fresh aggregates. The rebuild runs in a daemon thread to
    # avoid blocking the current search — worst case the query uses
    # the previous generation of aggregates, which is always safe.
    if getattr(cfg, "aggregate_enabled", False):
        from . import aggregates as _agg_module

        if _agg_module.should_auto_rebuild(palace_path):
            import threading

            threading.Thread(
                target=_agg_module.rebuild_dirty,
                kwargs={"palace_path": palace_path},
                daemon=True,
                name="mempalace.aggregates.rebuild_dirty",
            ).start()

    # Overfetch to give RRF room to work. We'll trim to n_results at the end.
    per_model_limit = max(n_results * 3, 15)

    def _run_one(slug: str) -> tuple[str, dict | None]:
        """Run a single-model query; return (slug, result_or_None).

        Errors are logged and translated to ``None`` so the fan-out
        merge loop can simply skip missing results.
        """
        try:
            sub = _hybrid_search_single(
                query,
                palace_path,
                keywords=keywords,
                keyword_mode=keyword_mode,
                since=since,
                until=until,
                as_of=as_of,
                wing=wing,
                room=room,
                n_results=per_model_limit,
                model=slug,
                enable_kg_ppr=enable_kg_ppr,
                aggregate_weights=effective_agg_weights,
            )
        except (OSError, chromadb.errors.ChromaError, ValueError, KeyError) as e:
            logger.warning("Fan-out: model %s failed: %s", slug, e)
            return (slug, None)
        if "error" in sub:
            logger.debug("Fan-out: model %s error: %s", slug, sub.get("error"))
            return (slug, None)
        return (slug, sub)

    # Cap the worker pool at the number of enabled models — going higher
    # wastes threads and creates lock contention in chromadb's SQLite
    # metadata store. Defensive ``getattr`` so test stubs that don't
    # carry the new config knob still work.
    max_workers = min(
        getattr(cfg, "fan_out_max_workers", 8),
        max(len(enabled), 1),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        per_model: dict[str, dict | None] = dict(executor.map(_run_one, enabled))

    drawer_weight = float(effective_agg_weights.get("drawer", 1.0))

    # Merge in the order of the enabled list so RRF ties resolve
    # deterministically regardless of which model finished first.
    fused: dict[str, dict] = {}
    for slug in enabled:
        sub = per_model.get(slug)
        if sub is None:
            continue

        for rank, hit in enumerate(sub.get("results", [])):
            drawer_id = hit.get("_drawer_id") or (hit["wing"], hit["room"], hit["source_file"])
            key = drawer_id if isinstance(drawer_id, str) else repr(drawer_id)
            contribution = drawer_weight * (1.0 / (_RRF_K + rank))

            existing = fused.get(key)
            if existing is None:
                fused[key] = {
                    "hit": dict(hit),
                    "score": contribution,
                    "source_models": [slug],
                }
            else:
                existing["score"] += contribution
                existing["source_models"].append(slug)
                # Prefer the first encountered hit's text/meta (earliest
                # rank wins); similarity stays whatever the first source
                # reported.

        # Second pass: fold this slug's aggregate-level boost into the
        # fused score. Boosts only land on drawers that are already in
        # the fused pool (i.e. Chroma returned them from at least one
        # model) — containers can re-rank drawers, never promote
        # drawers that the direct search never saw.
        agg_boost = sub.get("_agg_boost") or {}
        for did, boost in agg_boost.items():
            entry = fused.get(did)
            if entry is None:
                continue
            entry["score"] += float(boost)
            entry["hit"]["aggregate_boost"] = round(
                entry["hit"].get("aggregate_boost", 0.0) + float(boost), 6
            )

    # Sort by RRF score descending and take top ``n_results`` (which is
    # usually an overfetched limit from the outer ``hybrid_search`` —
    # the post-compression truncation happens in the caller so the
    # compression stage has headroom to collapse redundancy).
    ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:n_results]

    out_hits: list[dict] = []
    for entry in ordered:
        hit = entry["hit"]
        # Keep ``_drawer_id`` on the hit — the outer ``hybrid_search``
        # needs it for cluster-level metadata, and the final
        # ``_strip_internal_fields`` pass drops it on the way out.
        hit["rrf_score"] = round(entry["score"], 6)
        hit["source_models"] = entry["source_models"]
        out_hits.append(hit)

    return {
        "query": query,
        "filters": _filters_block(
            wing=wing,
            room=room,
            keywords=keywords,
            keyword_mode=keyword_mode,
            since=since,
            until=until,
            as_of=as_of,
            model="all",
        ),
        "results": out_hits,
        "fan_out": {
            "k": _RRF_K,
            "models_queried": enabled,
            "unique_drawers": len(fused),
        },
    }


def search_memories(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    *,
    model: str | None = None,
    compress: str = "auto",
    token_budget: int | None = None,
) -> dict:
    """
    Programmatic search — returns a dict instead of printing.
    Used by the MCP server and other callers that need data.

    Backward-compatible thin wrapper around :func:`hybrid_search` with no
    keyword / temporal constraints. Existing callers see the same response
    shape they always have (``{query, filters: {wing, room}, results}``) —
    the extra ``filters`` keys from ``hybrid_search`` are stripped here.

    The ``compress`` and ``token_budget`` args are accepted so the MCP
    tool surface can thread them through. On single-model queries with
    ``compress="auto"`` this resolves to ``"none"`` and the response is
    byte-for-byte unchanged from the pre-compression behavior.
    """
    result = hybrid_search(
        query,
        palace_path=palace_path,
        wing=wing,
        room=room,
        n_results=n_results,
        model=model,
        compress=compress,
        token_budget=token_budget,
    )
    if "error" in result:
        return result
    return {
        "query": result["query"],
        "filters": {"wing": wing, "room": room},
        "results": [
            {
                "text": h["text"],
                "wing": h["wing"],
                "room": h["room"],
                "source_file": h["source_file"],
                "similarity": h["similarity"],
            }
            for h in result["results"]
        ],
    }


def search(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    *,
    keywords: list | None = None,
    keyword_mode: str = "all",
    since: str | None = None,
    until: str | None = None,
    as_of: str | None = None,
    model: str | None = None,
    compress: str = "auto",
    token_budget: int | None = None,
    dup_threshold: float = 0.7,
    sent_threshold: float = 0.75,
    novelty_threshold: float = 0.2,
    rerank: str | None = None,
    rerank_prune: bool = True,
    enable_kg_ppr: bool = False,
    aggregate_weights: dict[str, float] | None = None,
):
    """CLI entry point — prints results, raises :class:`SearchError` on failure.

    Accepts the same keyword / temporal arguments as :func:`hybrid_search`
    plus ``model`` (pass ``"all"`` for fan-out across every enabled model),
    the full compression knob set (``compress``, ``token_budget``, the
    three Jaccard thresholds), the optional ``rerank`` + ``rerank_prune``
    reranker controls, and ``enable_kg_ppr`` for HippoRAG-style PPR over
    the knowledge graph.
    """
    result = hybrid_search(
        query,
        palace_path=palace_path,
        wing=wing,
        room=room,
        n_results=n_results,
        keywords=keywords,
        keyword_mode=keyword_mode,
        since=since,
        until=until,
        as_of=as_of,
        model=model,
        compress=compress,
        token_budget=token_budget,
        dup_threshold=dup_threshold,
        sent_threshold=sent_threshold,
        novelty_threshold=novelty_threshold,
        rerank=rerank,
        rerank_prune=rerank_prune,
        enable_kg_ppr=enable_kg_ppr,
        aggregate_weights=aggregate_weights,
    )

    if "error" in result:
        print(f"\n  {result['error']}")
        if result.get("hint"):
            print(f"  {result['hint']}")
        raise SearchError(result["error"])

    hits = result["results"]
    if not hits:
        print(f'\n  No results found for: "{query}"')
        return

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if model:
        print(f"  Model: {model}")
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    if keywords:
        print(f"  Keywords ({keyword_mode}): {', '.join(keywords)}")
    if since or until:
        print(f"  Filed between: {since or '...'} → {until or '...'}")
    if as_of:
        print(f"  As of: {as_of}")
    if result.get("fan_out"):
        fo = result["fan_out"]
        print(f"  Fan-out (RRF k={fo['k']}): {', '.join(fo['models_queried'])}")
    comp = result.get("compression")
    if (
        comp
        and comp.get("mode") != "none"
        and comp.get("input_hits", 0) > comp.get("output_hits", 0)
    ):
        print(
            f"  Compressed {comp['input_hits']} → {comp['output_hits']} hits "
            f"(clusters merged: {comp['clusters_merged']}, "
            f"sentences dropped: {comp['sentences_dropped']}, "
            f"{comp['input_tokens']} → {comp['output_tokens']} tokens, "
            f"{comp['ratio']}× denser)"
        )
        if comp.get("hits_gated_by_novelty"):
            print(f"  Novelty-gated: {comp['hits_gated_by_novelty']} hits dropped for low novelty")
        if comp.get("budget_reached"):
            print("  Token budget reached — remaining hits truncated")
    print(f"{'=' * 60}\n")

    for i, hit in enumerate(hits, 1):
        similarity = hit.get("similarity")
        print(f"  [{i}] {hit['wing']} / {hit['room']}")
        print(f"      Source: {hit['source_file']}")
        if hit.get("cluster_size", 1) > 1:
            merged_ids = hit.get("merged_drawer_ids", [])
            models = hit.get("merged_source_models", [])
            print(
                f"      Cluster:{hit['cluster_size']} "
                f"({len(merged_ids)} drawers across "
                f"{len(models)} models: {', '.join(models)})"
            )
        elif hit.get("source_models"):
            print(f"      From:   {', '.join(hit['source_models'])}")
        if hit.get("rrf_score") is not None:
            print(f"      RRF:    {hit['rrf_score']}")
        elif similarity is not None:
            print(f"      Match:  {similarity}")
        elif hit.get("filed_at"):
            print(f"      Filed:  {hit['filed_at']}")
        print()
        for line in (hit["text"] or "").strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()
