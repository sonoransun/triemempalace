"""
kg_ppr.py — HippoRAG-style Personalized PageRank over the knowledge graph.

**The architectural capstone of the retrieval expansion tranche.**
MemPalace is the rare RAG system that already has a proper temporal
knowledge graph sitting unused on the read path. HippoRAG (Gutiérrez
et al., NeurIPS 2024 — `arXiv 2405.14831
<https://arxiv.org/abs/2405.14831>`_) shows this graph is worth up to
+20% on multi-hop QA over non-graph baselines. This module makes the
KG a first-class retrieval signal by running Personalized PageRank
seeded on query entities and unioning the top-ranked drawers into
the normal search candidate set.

How it fits into the existing pipeline
--------------------------------------

The old pipeline was:

    query
      └─> trie keyword/temporal prefilter (candidate_ids)
          └─> Chroma vector query (ranks candidate_ids)
              └─> compress + rerank + response

The Tranche 5 pipeline (when ``enable_kg_ppr=True``):

    query
      ├─> trie keyword/temporal prefilter (trie_candidate_ids)
      └─> entity extraction (heuristic) → PPR over the KG
          └─> top-K entities → source_closet drawer IDs (kg_candidate_ids)
              └─> trie_candidate_ids ∪ kg_candidate_ids (unified set)
                  └─> Chroma vector query (ranks the unified set)
                      └─> compress + rerank + response

The KG contribution is **additive** — it brings new candidates into
the semantic query's ranked pool without dropping anything the trie
would have found. The Chroma vector query still does the final
semantic ranking, so drawers that look relevant to the query text
win regardless of whether they arrived via the trie or the PPR.

Why hand-rolled power iteration
-------------------------------

A typical personal palace's KG has 100-1000 triples. That's small
enough that a pure-Python power iteration over a sparse adjacency
dict runs in ~1 ms per query — faster than importing scipy. We
avoid the scipy dependency for the same reason we avoid torch in
the core install: zero new runtime deps is a core MemPalace value.

The implementation:

1. Load all triples into an ``{entity_id: {neighbor_id: weight}}``
   adjacency dict (cached per-process with a mtime check on the
   sqlite file so cold cache = one load + many queries).
2. Seed the PPR vector by placing mass on the query-extracted
   entity IDs. Un-seeded entities start at 0.
3. Iterate: ``v = damping * (M @ v) + (1 - damping) * seed`` where
   M is the normalized adjacency matrix.
4. Stop when the L1 distance between iterations < tol (usually
   < 10 iterations for a 500-triple graph).
5. Return ``{entity_id: score}`` sorted descending.

Graph construction notes
------------------------

* Triples are treated as **undirected** for PPR — the intuition is
  that if A works_at B, then a query about A should surface drawers
  about B and vice versa. This is what HippoRAG does.
* Edge weight = triple confidence (from the Tranche 4 voting rule).
  Higher-confidence triples propagate more mass.
* Expired triples (``valid_to`` set and in the past) are filtered
  out at graph-load time. If ``as_of`` is supplied, point-in-time
  filtering is applied the same way ``knowledge_graph.query_entity``
  does.
* Self-loops are dropped to avoid degenerate stationary
  distributions.

Cache invalidation: the loaded adjacency is re-built whenever the
sqlite file's mtime changes. This keeps the cache warm for read-
heavy workloads while picking up new triples added via
``mempalace kg-extract`` between queries.
"""

import json
import logging
import os
import sqlite3
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger("mempalace.kg_ppr")


# ── Cached adjacency ─────────────────────────────────────────────────


class _CachedGraph:
    """Thread-safe mtime-indexed cache of the KG adjacency structure.

    One instance per process. First call loads from sqlite; subsequent
    calls are a dict lookup + mtime check. Any change to the underlying
    sqlite file (new triple, confidence bump, invalidation) invalidates
    the cache on the next access.
    """

    def __init__(self):
        self._adj: dict[str, dict[str, float]] = {}
        self._entity_to_drawers: dict[str, list[str]] = {}
        self._mtime: float = -1.0
        self._db_path: str | None = None
        self._lock = threading.Lock()

    def get(self, db_path: str) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
        """Return ``(adjacency, entity_to_drawers)`` for the given KG path."""
        with self._lock:
            try:
                current_mtime = os.path.getmtime(db_path)
            except OSError:
                current_mtime = -1.0

            if self._db_path == db_path and self._mtime == current_mtime and self._adj:
                return self._adj, self._entity_to_drawers

            self._adj, self._entity_to_drawers = _build_adjacency(db_path)
            self._mtime = current_mtime
            self._db_path = db_path
            return self._adj, self._entity_to_drawers

    def clear(self) -> None:
        """Drop the cache — used by tests between temp KGs."""
        with self._lock:
            self._adj.clear()
            self._entity_to_drawers.clear()
            self._mtime = -1.0
            self._db_path = None


_graph_cache = _CachedGraph()


def clear_cache() -> None:
    """Public cache reset used by tests."""
    _graph_cache.clear()


# ── Graph construction ───────────────────────────────────────────────


def _build_adjacency(
    db_path: str,
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Read every currently-valid triple from sqlite and build:

    1. An undirected weighted adjacency dict keyed on entity_id:
       ``{entity_id: {neighbor_id: weight}}``. Weight is the triple's
       confidence score. Repeated edges between the same pair sum.

    2. An ``entity_to_drawers`` dict mapping each entity_id to the
       list of drawer_ids that evidence any triple involving that
       entity. Derived from the ``source_closet`` JSON list added in
       Tranche 4. Drawers never repeat in a single entity's list.

    Both structures are pure Python dicts — no numpy, no scipy. For
    KGs under ~10k triples this is trivially fast.
    """
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    entity_to_drawers: dict[str, set[str]] = defaultdict(set)

    if not os.path.exists(db_path):
        return {}, {}

    try:
        conn = sqlite3.connect(db_path, timeout=5)
    except sqlite3.Error as e:
        logger.warning("kg_ppr: cannot open KG at %s — %s", db_path, e)
        return {}, {}

    try:
        rows = conn.execute(
            "SELECT subject, object, confidence, source_closet FROM triples WHERE valid_to IS NULL"
        ).fetchall()
    except sqlite3.Error as e:
        logger.warning("kg_ppr: triples query failed — %s", e)
        conn.close()
        return {}, {}

    conn.close()

    for sub, obj, confidence, source_closet in rows:
        if not sub or not obj or sub == obj:
            continue  # skip self-loops
        weight = float(confidence or 0.5)

        adj[sub][obj] += weight
        adj[obj][sub] += weight

        if source_closet:
            drawers = _parse_source_closet(source_closet)
            for drawer_id in drawers:
                entity_to_drawers[sub].add(drawer_id)
                entity_to_drawers[obj].add(drawer_id)

    # Convert defaultdicts to plain dicts for the cached return value,
    # and sets to lists so the consumer can iterate deterministically.
    plain_adj: dict[str, dict[str, float]] = {k: dict(v) for k, v in adj.items()}
    plain_drawers: dict[str, list[str]] = {k: sorted(v) for k, v in entity_to_drawers.items()}
    return plain_adj, plain_drawers


def _parse_source_closet(raw: str) -> list[str]:
    """Parse the ``source_closet`` column — handles both the
    Tranche-4 JSON-list format and the legacy bare-string format.

    Defensive: any parse failure returns ``[raw]`` so a malformed
    legacy row still contributes its one drawer_id to the graph.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [raw]


# ── Personalized PageRank ────────────────────────────────────────────


def personalized_pagerank(
    seed_entities: list[str],
    *,
    kg_db_path: str,
    damping: float = 0.85,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Run Personalized PageRank seeded on ``seed_entities``.

    Parameters
    ----------
    seed_entities:
        List of entity IDs (already lowercased + normalized via the
        KG's ``_entity_id`` helper). The seed vector places uniform
        mass across these. Entities not in the graph are silently
        dropped from the seed.
    kg_db_path:
        Path to the ``knowledge_graph.sqlite3`` file.
    damping:
        PageRank damping factor. 0.85 is the Page/Brin default and
        also what HippoRAG uses.
    max_iter:
        Hard cap on iterations. Typical convergence is <10 iterations
        for personal-scale KGs.
    tol:
        L1 convergence threshold. Iteration stops when the change
        between steps drops below this.

    Returns
    -------
    dict[entity_id, score]
        Scores normalized to sum to 1. Sorted descending by caller if
        desired. An empty dict is returned when the KG is empty or
        no seeds land in the graph (callers should fall back to the
        normal search path in that case).
    """
    adj, _ = _graph_cache.get(kg_db_path)
    if not adj:
        return {}

    # Normalize seed: keep only seeds that exist in the graph.
    valid_seeds = [s for s in seed_entities if s in adj]
    if not valid_seeds:
        return {}

    # Seed vector: uniform over valid seeds, zero elsewhere.
    seed_vec: dict[str, float] = {}
    seed_weight = 1.0 / len(valid_seeds)
    for s in valid_seeds:
        seed_vec[s] = seed_weight

    # Initialize scores at the seed distribution.
    scores: dict[str, float] = dict(seed_vec)

    # Pre-compute the row-normalized transition weights so the
    # iteration is a simple sum.
    out_weights: dict[str, float] = {
        node: sum(neighbors.values()) for node, neighbors in adj.items()
    }

    for _ in range(max_iter):
        new_scores: dict[str, float] = {}

        # Random walk contribution: mass flowing from each node to its
        # neighbors, weighted by edge weight / row sum.
        for node, current_score in scores.items():
            total_out = out_weights.get(node, 0.0)
            if total_out <= 0:
                # Dangling node — push mass straight to the seed.
                for s, w in seed_vec.items():
                    new_scores[s] = new_scores.get(s, 0.0) + current_score * w
                continue
            for neighbor, edge_weight in adj[node].items():
                contribution = current_score * (edge_weight / total_out)
                new_scores[neighbor] = new_scores.get(neighbor, 0.0) + contribution

        # Combine walk (damping) with teleport (1 - damping) back to seed.
        combined: dict[str, float] = {}
        for node, walk_score in new_scores.items():
            combined[node] = damping * walk_score
        for seed_node, seed_weight in seed_vec.items():
            combined[seed_node] = combined.get(seed_node, 0.0) + (1.0 - damping) * seed_weight

        # L1 convergence check
        delta = 0.0
        all_keys = set(combined) | set(scores)
        for k in all_keys:
            delta += abs(combined.get(k, 0.0) - scores.get(k, 0.0))
        scores = combined
        if delta < tol:
            break

    return scores


# ── Query entity extraction ──────────────────────────────────────────


def extract_query_entities(query: str) -> list[str]:
    """Extract candidate entity names from a query for seeding PPR.

    Reuses the same capitalization-based heuristic as
    ``kg_extract.HeuristicExtractor`` — proper nouns (capitalized
    words, possibly multi-word) are treated as potential entity
    names. This is intentionally loose because the downstream graph
    lookup will filter to only names that actually exist in the KG.

    Returns a list of **normalized entity IDs** (lowercased +
    spaces-to-underscores), matching the format used by
    ``knowledge_graph._entity_id``. The caller can pass this
    directly to :func:`personalized_pagerank` as the seed list.
    """
    import re

    if not query:
        return []

    # Match proper nouns: 1-3 capitalized words. Allow apostrophes
    # and hyphens within a word so "O'Brien" and "Jean-Luc" resolve.
    pattern = re.compile(r"\b[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2}\b")
    names = pattern.findall(query)

    # Normalize to KG entity_id format
    seen: set[str] = set()
    seeds: list[str] = []
    for name in names:
        entity_id = name.lower().replace(" ", "_").replace("'", "")
        if entity_id not in seen:
            seeds.append(entity_id)
            seen.add(entity_id)
    return seeds


# ── Candidate fusion helper ──────────────────────────────────────────


def kg_ppr_candidates(
    query: str,
    *,
    kg_db_path: str,
    top_k: int = 20,
    damping: float = 0.85,
) -> dict[str, Any]:
    """One-shot helper that extracts query entities, runs PPR, and
    returns the fused drawer-ID candidate set.

    This is what ``searcher.hybrid_search`` calls when
    ``enable_kg_ppr=True``. The return envelope is a dict so the
    caller can surface the debugging information (seeds, top
    entities, contributing triples) in the search response for
    transparency.

    Returns
    -------
    dict
        ``{
            "seeds": list[str],              # query entities found in KG
            "top_entities": list[(id, score)], # top K by PPR score
            "drawer_ids": set[str],          # union of drawer_ids for top entities
            "skipped_reason": str | None,    # "no_kg" / "no_seeds" / None
        }``
    """
    envelope: dict[str, Any] = {
        "seeds": [],
        "top_entities": [],
        "drawer_ids": set(),
        "skipped_reason": None,
    }

    query_seeds = extract_query_entities(query)
    if not query_seeds:
        envelope["skipped_reason"] = "no_entities_in_query"
        return envelope

    adj, entity_to_drawers = _graph_cache.get(kg_db_path)
    if not adj:
        envelope["skipped_reason"] = "no_kg"
        return envelope

    valid_seeds = [s for s in query_seeds if s in adj]
    envelope["seeds"] = valid_seeds
    if not valid_seeds:
        envelope["skipped_reason"] = "no_seeds_in_graph"
        return envelope

    scores = personalized_pagerank(
        valid_seeds,
        kg_db_path=kg_db_path,
        damping=damping,
    )
    if not scores:
        envelope["skipped_reason"] = "ppr_returned_empty"
        return envelope

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    envelope["top_entities"] = [(eid, round(score, 6)) for eid, score in top]

    drawer_ids: set[str] = set()
    for entity_id, _score in top:
        for drawer_id in entity_to_drawers.get(entity_id, []):
            drawer_ids.add(drawer_id)
    envelope["drawer_ids"] = drawer_ids
    return envelope


__all__ = [
    "personalized_pagerank",
    "extract_query_entities",
    "kg_ppr_candidates",
    "clear_cache",
]
