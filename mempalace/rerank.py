"""
rerank.py — Cross-encoder reranker registry with optional per-token pruning.

MemPalace's retrieval hot path is currently:

    query → trie prefilter → Chroma vector search → (optional) compression

This module adds an optional **reranker** stage between the Chroma
search and the compression pass. Cross-encoder rerankers read each
(query, candidate) pair jointly through a small transformer, producing
a far more accurate similarity score than the bi-encoder dense search
can on its own. Published work routinely shows +5-10 nDCG points on
BEIR-scale benchmarks.

Two backends ship as optional pip extras:

* **Provence** (`naver/provence-reranker-debertav3-v1`, ICLR 2025) —
  a single DeBERTa-v3 model that jointly reranks *and* emits per-token
  keep/drop labels, pruning irrelevant sentences from each candidate.
  Install with ``pip install 'mempalace[rerank-provence]'``. Requires
  ``transformers + torch``. Use ``prune=True`` (default) to write the
  pruned text back into the hit's ``pruned_text`` field; downstream
  ``compress.py`` will prefer ``pruned_text`` over ``text`` if present.
* **BGE-reranker-v2-m3** (`BAAI/bge-reranker-v2-m3`) — the standard
  pure-rerank cross-encoder. Loaded via fastembed's ONNX
  ``TextCrossEncoder`` path, so no torch required. Install with
  ``pip install 'mempalace[rerank-bge]'``.

Both rerankers:

* Are **optional** — no extra, no import, zero cost to core install.
* **Lazy-load** weights on first ``__call__``, not at module import.
* Are **singletons** per process via ``_loaded_rerankers[slug]``.
* Implement the same ``rerank(query, hits, top_k=None, ...) -> list[hit]``
  interface so callers are backend-agnostic.

The registry mirrors the ``embeddings.py`` pattern deliberately —
future rerankers can be added with a single ``register()`` call and
an adapter class. Users discover what's available via
``mempalace rerankers list`` or the ``mempalace_list_rerankers`` MCP
tool.

Architecture notes
------------------

* **Where reranking happens in the pipeline**:
  ``searcher.hybrid_search`` — after the trie prefilter and the
  Chroma vector query, before ``compress.compress_results``. Only the
  trie+Chroma path is reranked; pure-keyword/temporal queries (those
  without a ``query_texts=``) skip reranking entirely because the
  input is already a bitmap, not a ranked list.
* **Pruning semantics**: Provence sets ``hit["pruned_text"]``
  alongside (not replacing) ``hit["text"]``. The compression stage
  checks for ``pruned_text`` first and falls back to ``text``. This
  way the original verbatim content is preserved in the hit envelope
  even when a pruned version is used for compression math.
* **No-op fallback**: if ``rerank=None`` (the default) nothing in
  this module runs — ``searcher.hybrid_search`` keeps its current
  behavior exactly.
"""

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("mempalace.rerank")


# ── Spec + registry ───────────────────────────────────────────────────


@dataclass(frozen=True)
class RerankSpec:
    """Static description of a reranker model.

    Mirrors :class:`mempalace.embeddings.EmbeddingSpec` — specs are
    frozen metadata; factories are dispatched per-backend inside
    :func:`load_reranker`.
    """

    slug: str
    display_name: str
    description: str
    backend: str  # "transformers-provence" | "fastembed-bge"
    model_id: str
    max_length: int  # max tokens per (query, candidate) pair
    supports_pruning: bool  # True iff the model emits per-token labels
    extras_required: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, RerankSpec] = {}


def register(spec: RerankSpec) -> None:
    """Add a reranker spec to the process-wide registry."""
    REGISTRY[spec.slug] = spec


def list_reranker_specs() -> list[RerankSpec]:
    """Return every registered spec in registration order."""
    return list(REGISTRY.values())


def get_reranker_spec(slug: str) -> RerankSpec:
    """Look up a spec by slug. Raises ``KeyError`` on unknown slug."""
    try:
        return REGISTRY[slug]
    except KeyError as e:
        available = ", ".join(sorted(REGISTRY.keys())) or "(none)"
        raise KeyError(f"Unknown reranker {slug!r}. Available: {available}") from e


def is_installed(spec: RerankSpec) -> bool:
    """Return True iff every required extra is importable.

    Uses ``importlib.util.find_spec`` for a zero-cost probe — never
    triggers an actual import, so the check is safe in hot paths.
    """
    return all(importlib.util.find_spec(extra) is not None for extra in spec.extras_required)


# ── Reranker protocol + adapters ──────────────────────────────────────


class Reranker(Protocol):
    """Common reranker interface.

    All adapters take a query + list of hits and return a new list
    with ``rerank_score`` set on every hit. The Provence adapter
    additionally sets ``pruned_text`` and ``pruning_stats`` when
    ``prune=True``.
    """

    def rerank(
        self,
        query: str,
        hits: list[dict],
        *,
        top_k: int | None = None,
        prune: bool = True,
    ) -> list[dict]:
        """Rerank the hits by (query, candidate) cross-encoder score.

        ``top_k=None`` keeps every hit in reranked order. ``top_k=N``
        truncates to the top N.

        ``prune`` is only meaningful for adapters whose spec has
        ``supports_pruning=True``; others ignore it.
        """


class _MissingExtrasError(RuntimeError):
    """Raised when a reranker's optional extras aren't installed."""


def _require_extras(spec: RerankSpec) -> None:
    if not is_installed(spec):
        extras = ", ".join(spec.extras_required)
        extra_name = "rerank-provence" if spec.backend == "transformers-provence" else "rerank-bge"
        raise _MissingExtrasError(
            f"Reranker {spec.slug!r} requires the {extras!r} package(s). "
            f"Install with:  pip install 'mempalace[{extra_name}]'"
        )


# Process-wide cache. Keyed on slug so every caller in the same process
# shares one loaded model. Populated lazily on first ``load_reranker``.
_loaded_rerankers: dict[str, Any] = {}


def load_reranker(slug: str) -> Any:
    """Return a Reranker adapter for the given slug.

    **Opportunistic load gate.** The backend module is imported only
    here. The underlying model weights are loaded even later, on the
    first ``rerank()`` call.

    Cached: repeated calls with the same slug return the same instance.
    """
    if slug in _loaded_rerankers:
        return _loaded_rerankers[slug]

    spec = get_reranker_spec(slug)
    _require_extras(spec)

    if spec.backend == "transformers-provence":
        adapter: Any = _ProvenceReranker(spec)
    elif spec.backend == "fastembed-bge":
        adapter = _BgeReranker(spec)
    else:
        raise ValueError(f"Unknown reranker backend {spec.backend!r} for {slug!r}")

    _loaded_rerankers[slug] = adapter
    return adapter


def clear_cache() -> None:
    """Drop every cached reranker adapter. Useful for tests."""
    _loaded_rerankers.clear()


# ── Provence adapter: unified rerank + per-token pruning ──────────────


class _ProvenceReranker:
    """Provence reranker + context pruner (Chirkova et al., ICLR 2025).

    Provence's value proposition: one DeBERTa-v3 forward pass per
    (query, candidate) produces *both* a relevance score *and* a
    per-token keep/drop label over the candidate. Drops ~99% of
    off-topic sentences while keeping ~80-90% of relevant text. This
    lets MemPalace unify reranking and context pruning into a single
    stage, saving a second model pass.

    Runtime cost: ~400M params, CPU-feasible via the transformers
    library. First call loads the tokenizer + model (~1-3 s on a
    warm cache); subsequent calls are pure forward passes.
    """

    def __init__(self, spec: RerankSpec):
        self._spec = spec
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        transformers = importlib.import_module("transformers")
        logger.info("Loading Provence reranker %s", self._spec.model_id)
        # The published Provence checkpoint exposes a custom class via
        # ``trust_remote_code=True``. This is the same pattern the
        # naver/provence-reranker-debertav3-v1 model card uses.
        self._model = transformers.AutoModel.from_pretrained(
            self._spec.model_id, trust_remote_code=True
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._spec.model_id)

    def rerank(
        self,
        query: str,
        hits: list[dict],
        *,
        top_k: int | None = None,
        prune: bool = True,
    ) -> list[dict]:
        """Rerank hits via Provence. Optionally prune each candidate.

        Returns a new list of dicts. Each hit gains:
            ``rerank_score`` — float, higher is more relevant
            ``pruned_text`` — string, only present when ``prune=True``
            ``pruning_stats`` — {"chars_in": N, "chars_out": M, "ratio": r}
        """
        if not hits:
            return hits
        self._ensure()

        # Provence's .process() API takes a question + list of contexts
        # and returns reranked contexts with per-token labels applied.
        # The exact signature is {question, context} per the model card.
        assert self._model is not None  # narrow for mypy
        try:
            processed = self._model.process(
                question=query,
                context=[h.get("text", "") for h in hits],
            )
        except (AttributeError, TypeError, RuntimeError) as e:
            # Broad but targeted: AttributeError guards against model
            # API drift, TypeError against signature changes,
            # RuntimeError against inference failures. Fall back to
            # the unranked list with a warning rather than crashing
            # the search path.
            logger.warning("Provence rerank failed: %s — returning unranked", e)
            return hits

        # ``processed`` shape per model card:
        #   {"reranking_score": list[float], "pruned_context": list[str]}
        scores = processed.get("reranking_score", [0.0] * len(hits))
        pruned = processed.get("pruned_context", [h.get("text", "") for h in hits])

        out: list[dict] = []
        for hit, score, pruned_text in zip(hits, scores, pruned, strict=False):
            new_hit = dict(hit)
            new_hit["rerank_score"] = float(score)
            if prune:
                chars_in = len(hit.get("text", ""))
                chars_out = len(pruned_text)
                new_hit["pruned_text"] = pruned_text
                new_hit["pruning_stats"] = {
                    "chars_in": chars_in,
                    "chars_out": chars_out,
                    "ratio": round(chars_out / max(chars_in, 1), 3),
                }
            out.append(new_hit)

        out.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out

    def name(self) -> str:
        return f"provence:{self._spec.slug}"


# ── BGE adapter: pure rerank, ONNX via fastembed ──────────────────────


class _BgeReranker:
    """BGE-reranker-v2-m3 via fastembed's cross-encoder path.

    fastembed ships ONNX-exported cross-encoders; no torch dependency.
    Pure rerank — no per-token pruning — so downstream compression
    flows through ``compress.compress_results`` unchanged.
    """

    def __init__(self, spec: RerankSpec):
        self._spec = spec
        self._model: Any | None = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        fastembed = importlib.import_module("fastembed")
        logger.info("Loading BGE reranker %s", self._spec.model_id)
        # fastembed exposes TextCrossEncoder for reranker-style models.
        # Falls back to a generic class if the version predates
        # reranker support; in that case ``_require_extras`` already
        # confirmed fastembed is importable but the feature may be
        # missing. We surface that as an explicit ImportError.
        if not hasattr(fastembed, "TextCrossEncoder"):
            raise ImportError(
                "fastembed >= 0.3.5 is required for reranker support; "
                "upgrade with:  pip install -U 'mempalace[rerank-bge]'"
            )
        self._model = fastembed.TextCrossEncoder(model_name=self._spec.model_id)

    def rerank(
        self,
        query: str,
        hits: list[dict],
        *,
        top_k: int | None = None,
        prune: bool = True,
    ) -> list[dict]:
        """Rerank hits by BGE cross-encoder score.

        ``prune`` is ignored — BGE has no per-token labels. The
        argument exists so callers can pass ``prune=True`` as their
        default without needing to know the backend's capabilities.
        """
        if not hits:
            return hits
        self._ensure()

        texts = [h.get("text", "") for h in hits]
        assert self._model is not None  # narrow for mypy
        try:
            # fastembed's TextCrossEncoder takes a query and list of docs;
            # returns an iterable of floats (one per doc).
            scores = list(self._model.rerank(query, texts))
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning("BGE rerank failed: %s — returning unranked", e)
            return hits

        out: list[dict] = []
        for hit, score in zip(hits, scores, strict=False):
            new_hit = dict(hit)
            new_hit["rerank_score"] = float(score)
            out.append(new_hit)

        out.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out

    def name(self) -> str:
        return f"bge-reranker:{self._spec.slug}"


# ── Built-in registry seed ────────────────────────────────────────────


def _register_builtins() -> None:
    """Populate the registry with the two shipping rerankers.

    Called once at module import. Idempotent — re-calling rebuilds
    the same entries.
    """
    register(
        RerankSpec(
            slug="provence",
            display_name="Provence (DeBERTa-v3 rerank + prune)",
            description=(
                "Unified cross-encoder reranker and per-token context "
                "pruner from Chirkova et al., ICLR 2025. One DeBERTa-v3 "
                "forward pass per (query, candidate) pair produces both "
                "a relevance score and per-token keep/drop labels, "
                "dropping ~99% of off-topic sentences while preserving "
                "~80-90% of relevant text. ~400M params; CPU-feasible."
            ),
            backend="transformers-provence",
            model_id="naver/provence-reranker-debertav3-v1",
            max_length=512,
            supports_pruning=True,
            extras_required=("transformers", "torch"),
        )
    )
    register(
        RerankSpec(
            slug="bge",
            display_name="BGE-reranker-v2-m3 (multilingual cross-encoder)",
            description=(
                "BAAI's multilingual cross-encoder reranker. 568M params "
                "but ships as ONNX via fastembed — no torch required. "
                "Pure rerank: emits a single relevance score per "
                "(query, candidate) pair. Pair with MemPalace's existing "
                "compress.py pipeline for context compression."
            ),
            backend="fastembed-bge",
            model_id="BAAI/bge-reranker-v2-m3",
            max_length=8192,
            supports_pruning=False,
            extras_required=("fastembed",),
        )
    )


_register_builtins()
