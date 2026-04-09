"""
compress.py — Aggregate + compress multi-model fan-out search results.

The cross-model fan-out search (``searcher._hybrid_search_fan_out``)
runs a query against every enabled embedding model's Chroma collection
and merges the result lists via Reciprocal Rank Fusion. That already
dedupes by ``_drawer_id``, but it does **not** collapse drawers that
share most of their content without sharing an ID — which is exactly
what happens when the same underlying content was mined into multiple
model collections as separately-hashed drawers.

This module is the result-set compression pipeline that runs over the
merged hit list and produces a collapsed, non-redundant output.

Four modes, progressively more aggressive:

* ``"none"`` — passthrough. Returns the input list with a zero-stats
  envelope. Byte-for-byte backward compatible.
* ``"dedupe"`` — drawer-level clustering by trigram Jaccard similarity
  over the ``trie_index.tokenize``-normalized token stream. Each
  cluster's members are merged under a single representative hit;
  non-representatives land in the hit's ``variants`` list.
* ``"sentences"`` — everything ``dedupe`` does, plus sentence-level
  shingle dedupe within the cluster representatives. Repeated sentences
  that already appeared in an earlier hit are dropped.
* ``"aggressive"`` — everything ``sentences`` does, plus a novelty gate
  (drop hits contributing < ``novelty_threshold`` new trigrams vs the
  accumulated context) and an optional token-budget enforcement that
  halts ingestion once the cumulative output exceeds the budget.

Reuse-only, zero new deps
-------------------------

* :func:`mempalace.trie_index.tokenize` does the lowercased,
  stopword-filtered, identifier-aware tokenization. Reusing it means the
  trie keyword index and this compression pipeline agree on what "same
  content" means.
* :func:`mempalace.dialect.Dialect.count_tokens` is the authoritative
  word-based × 1.3 token counter used everywhere else in the codebase.
* Sentence splitting is a single regex (``[.!?\\n]+``) copied verbatim
  from ``dialect._extract_key_sentence`` so the two modules agree on
  sentence boundaries.

Nothing else is imported at module scope. The compression pipeline is
safe to import from a cold interpreter without touching any model
weights or network resources.
"""

import re

from .dialect import Dialect
from .trie_index import tokenize

# ── Constants ─────────────────────────────────────────────────────────

# Valid compression modes, in order of aggressiveness.
#
# ``llmlingua2`` is a learned compression mode (Pan et al., ACL 2024 —
# arXiv 2403.12968) that uses a small bidirectional transformer
# (xlm-roberta-large) to label each token keep/drop with targets
# distilled from GPT-4. Drops 2-5× more tokens than the heuristic
# ``aggressive`` mode at the same preserved-info level. Optional
# extra: ``pip install 'mempalace[compress-llmlingua]'``.
MODES = ("none", "dedupe", "sentences", "aggressive", "llmlingua2")

# Sentence splitter. Matches ``dialect._extract_key_sentence`` so the
# two modules produce identical segmentation.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")

# Lazy singleton for the LLMLingua-2 compressor. Loaded on first
# ``llmlingua2`` call. Import cost is ~500 MB model download + ~1-2s
# tokenizer setup; amortized across all subsequent calls.
_llmlingua_compressor = None


# ── Public API ────────────────────────────────────────────────────────


def compress_results(
    hits: list[dict],
    *,
    mode: str = "dedupe",
    token_budget: int | None = None,
    dup_threshold: float = 0.7,
    sent_threshold: float = 0.75,
    novelty_threshold: float = 0.2,
) -> tuple[list[dict], dict]:
    """Compress a list of search hits into a collapsed, non-redundant set.

    Parameters
    ----------
    hits:
        Ordered list of hit dicts from the searcher. Each hit should
        have at minimum ``text``, and optionally ``rrf_score``,
        ``similarity``, ``source_models``, ``_drawer_id``.
    mode:
        Compression level. See module docstring.
    token_budget:
        Optional maximum output tokens. Only honored in
        ``mode="aggressive"``. ``None`` means no budget.
    dup_threshold:
        Drawer-level trigram-Jaccard cutoff for the dedupe clustering
        pass. Default 0.7 matches the Broder near-duplicate cut.
    sent_threshold:
        Sentence-level bigram-Jaccard cutoff. Bigrams are more forgiving
        than trigrams at the sentence length. Default 0.75.
    novelty_threshold:
        Minimum fraction of trigrams in a hit's text that must be
        *novel* (not seen in any earlier kept hit) to survive the
        aggressive-mode novelty gate. Default 0.2.

    Returns
    -------
    tuple
        ``(compressed_hits, stats)``. ``compressed_hits`` is the
        collapsed list in the same rank order as the input.
        ``stats`` is a dict describing the compression outcome; see
        :func:`_empty_stats` for the shape.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown compression mode {mode!r}. Valid: {MODES}")

    # ── Mode: none — pure passthrough ─────────────────────────────────
    if mode == "none" or not hits:
        stats = _empty_stats(mode)
        if hits:
            input_tokens = sum(Dialect.count_tokens(h.get("text") or "") for h in hits)
            stats["input_hits"] = len(hits)
            stats["output_hits"] = len(hits)
            stats["input_tokens"] = input_tokens
            stats["output_tokens"] = input_tokens
            stats["ratio"] = 1.0
        return list(hits), stats

    # ── Mode: llmlingua2 — learned token-level compression ───────────
    # Dispatches to a separate helper because the algorithm has no
    # overlap with the heuristic dedupe / sentences / aggressive
    # pipeline. The helper gracefully falls back to "none" when the
    # optional extra isn't installed, so callers can always request
    # "llmlingua2" without crashing on a bare install.
    if mode == "llmlingua2":
        return _compress_llmlingua2(hits, token_budget=token_budget)

    # ── Pass A: fingerprint every hit (unigram token set) ────────────
    fingerprints = [_drawer_fingerprint(h.get("text") or "") for h in hits]
    input_tokens = sum(Dialect.count_tokens(h.get("text") or "") for h in hits)

    # ── Pass B: cluster by Jaccard similarity ─────────────────────────
    clusters = _cluster_by_jaccard(fingerprints, dup_threshold)

    # Build representatives: for each cluster, pick the best hit and
    # attach the rest as variants.
    representatives: list[dict] = []
    for member_idxs in clusters:
        rep_idx = _pick_representative(hits, member_idxs)
        rep = dict(hits[rep_idx])  # shallow copy so we can mutate

        merged_ids: list[str] = []
        merged_models: list[str] = []
        variants: list[dict] = []
        for idx in member_idxs:
            src = hits[idx]
            did = src.get("_drawer_id") or src.get("drawer_id") or ""
            if did:
                merged_ids.append(did)
            for m in src.get("source_models") or []:
                if m not in merged_models:
                    merged_models.append(m)
            if idx != rep_idx:
                variants.append(
                    {
                        "drawer_id": did,
                        "text": src.get("text") or "",
                        "similarity": src.get("similarity"),
                        "source_model": (src.get("source_models") or [None])[0],
                    }
                )

        rep["merged_drawer_ids"] = merged_ids
        rep["merged_source_models"] = merged_models or list(rep.get("source_models") or [])
        rep["cluster_size"] = len(member_idxs)
        rep["variants"] = variants
        original = rep.get("text") or ""
        rep["original_tokens"] = Dialect.count_tokens(original)
        rep["output_tokens"] = rep["original_tokens"]
        representatives.append(rep)

    clusters_merged = sum(1 for c in clusters if len(c) > 1)

    # ── Pass C: sentence-level dedupe inside representatives ──────────
    sentences_dropped = 0
    if mode in ("sentences", "aggressive"):
        representatives, sentences_dropped = _dedupe_sentences(
            representatives, sent_threshold=sent_threshold
        )

    # ── Pass D: novelty gate + token budget (aggressive only) ────────
    hits_gated_by_novelty = 0
    budget_reached = False
    if mode == "aggressive":
        representatives, hits_gated_by_novelty, budget_reached = _apply_novelty_and_budget(
            representatives,
            novelty_threshold=novelty_threshold,
            token_budget=token_budget,
        )

    output_tokens = sum(
        h.get("output_tokens", Dialect.count_tokens(h.get("text") or "")) for h in representatives
    )

    stats = _empty_stats(mode)
    stats["input_hits"] = len(hits)
    stats["output_hits"] = len(representatives)
    stats["input_tokens"] = input_tokens
    stats["output_tokens"] = output_tokens
    stats["ratio"] = round(input_tokens / output_tokens, 3) if output_tokens else 0.0
    stats["clusters_merged"] = clusters_merged
    stats["sentences_dropped"] = sentences_dropped
    stats["hits_gated_by_novelty"] = hits_gated_by_novelty
    stats["budget_reached"] = budget_reached

    return representatives, stats


# ── Private helpers ───────────────────────────────────────────────────


def _empty_stats(mode: str) -> dict:
    """Zero-valued stats dict shell."""
    return {
        "mode": mode,
        "input_hits": 0,
        "output_hits": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "ratio": 1.0,
        "clusters_merged": 0,
        "sentences_dropped": 0,
        "hits_gated_by_novelty": 0,
        "budget_reached": False,
    }


def _drawer_fingerprint(text: str) -> set[str]:
    """Token-set fingerprint for drawer-level near-duplicate detection.

    We use **unigram** (token set) Jaccard here, not trigrams. Trigrams
    are the textbook choice for web-scale documents, but drawers in
    MemPalace are typically short (dozens of tokens) and a single word
    swap tanks trigram Jaccard — a one-word paraphrase between two
    otherwise-identical sentences drops trigram overlap from 1.0 to
    ~0.4, below any reasonable threshold.

    Token set Jaccard is more forgiving to word swaps while still
    rejecting unrelated content: two drawers with wildly different
    vocabularies can't both score above 0.7 on the unigram metric.
    The trie tokenizer already drops stopwords and lowercases
    identifiers, so the resulting set is a content fingerprint
    rather than a bag-of-words.
    """
    return set(tokenize(text))


def _trigram_shingles(text: str) -> set[tuple[str, str, str]]:
    """Trigram set. Used by the aggressive-mode novelty gate where we
    want to track *structural* novelty (sentence order, phrasing) and
    not just vocabulary.
    """
    toks = tokenize(text)
    if len(toks) < 3:
        return {(t, "", "") for t in toks}
    return {(toks[i], toks[i + 1], toks[i + 2]) for i in range(len(toks) - 2)}


def _bigram_shingles(text: str) -> set[tuple[str, str]]:
    """Bigram shingles for sentence-level comparison.

    Sentences are shorter than drawers, so bigrams give a denser
    similarity signal than trigrams would at the same length.
    """
    toks = tokenize(text)
    if len(toks) < 2:
        return {(t, "") for t in toks}
    return {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}


def _jaccard(a: set, b: set) -> float:
    """Standard Jaccard similarity. Returns 0.0 for two empty sets."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cluster_by_jaccard(fingerprints: list[set], threshold: float) -> list[list[int]]:
    """Single-linkage cluster hits whose fingerprints exceed the Jaccard cutoff.

    Uses union-find to amortize the pairwise comparisons into
    near-linear time. Returns a list of clusters in the order the
    representative members first appear in the input — rank order is
    preserved so downstream consumers still see the highest-scoring
    results first.
    """
    n = len(fingerprints)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(fingerprints[i], fingerprints[j]) >= threshold:
                union(i, j)

    # Group indices by root, preserving first-seen order of the roots.
    seen_roots: dict[int, list[int]] = {}
    ordered_roots: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in seen_roots:
            seen_roots[root] = []
            ordered_roots.append(root)
        seen_roots[root].append(i)

    return [seen_roots[r] for r in ordered_roots]


def _pick_representative(hits: list[dict], member_idxs: list[int]) -> int:
    """Pick the best member of a cluster to serve as the representative.

    Priority:
      1. Highest ``rrf_score`` (fan-out path)
      2. Highest ``similarity`` (single-model path)
      3. Longest ``text`` (tie-breaker: more information per drawer)
    """

    def score(idx: int) -> tuple[float, float, int]:
        h = hits[idx]
        rrf = float(h.get("rrf_score") or 0.0)
        sim = float(h.get("similarity") or 0.0)
        tlen = len(h.get("text") or "")
        return (rrf, sim, tlen)

    return max(member_idxs, key=score)


def _dedupe_sentences(
    representatives: list[dict], *, sent_threshold: float
) -> tuple[list[dict], int]:
    """Walk rep sentences in order, drop those overlapping earlier ones.

    Uses bigram Jaccard over per-sentence shingle sets. Sentences are
    segmented with the module-level ``_SENTENCE_SPLIT_RE``. Joining
    survivors uses ``". "`` — the original punctuation is lost but the
    semantic content and sentence order are preserved.

    Representatives that lose every sentence are removed from the
    output list entirely.
    """
    seen_shingles: set[tuple[str, str]] = set()
    out_reps: list[dict] = []
    total_dropped = 0

    for rep in representatives:
        raw = rep.get("text") or ""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(raw) if s.strip()]
        if not sentences:
            out_reps.append(rep)
            continue

        kept: list[str] = []
        dropped = 0
        for sent in sentences:
            shingles = _bigram_shingles(sent)
            if not shingles:
                kept.append(sent)
                continue
            is_dup = False
            if seen_shingles:
                overlap = len(shingles & seen_shingles)
                union = len(shingles | seen_shingles)
                if union and (overlap / len(shingles)) >= sent_threshold:
                    # Asymmetric: if most of this sentence's bigrams are
                    # already present globally, drop it. This is more
                    # aggressive than pure Jaccard and better matches
                    # the "this sentence is already represented" intent.
                    is_dup = True
            if is_dup:
                dropped += 1
            else:
                kept.append(sent)
                seen_shingles.update(shingles)

        total_dropped += dropped
        if kept:
            new_rep = dict(rep)
            new_rep["text"] = ". ".join(kept)
            new_rep["output_tokens"] = Dialect.count_tokens(new_rep["text"])
            out_reps.append(new_rep)
        # else: rep had all sentences dropped → skip it entirely

    return out_reps, total_dropped


def _apply_novelty_and_budget(
    representatives: list[dict],
    *,
    novelty_threshold: float,
    token_budget: int | None,
) -> tuple[list[dict], int, bool]:
    """Aggressive-mode final pass.

    * **Novelty gate**: for each representative in rank order, compute
      the fraction of its trigrams that aren't in the accumulated
      context. Drop if below ``novelty_threshold``. Increment the
      gate counter.
    * **Token budget**: if set, maintain a running token total. When
      adding the next rep would push the total over the budget, stop
      and set ``budget_reached=True``.

    Returns ``(kept, gated_count, budget_reached)``.
    """
    kept: list[dict] = []
    seen_trigrams: set[tuple[str, str, str]] = set()
    gated = 0
    budget_reached = False
    running_tokens = 0

    for rep in representatives:
        text = rep.get("text") or ""
        trigrams = _trigram_shingles(text)

        # Novelty gate
        if trigrams and seen_trigrams:
            novel = trigrams - seen_trigrams
            novelty = len(novel) / len(trigrams)
            if novelty < novelty_threshold:
                gated += 1
                continue

        # Token budget
        rep_tokens = rep.get("output_tokens") or Dialect.count_tokens(text)
        if token_budget is not None and running_tokens + rep_tokens > token_budget:
            budget_reached = True
            break

        kept.append(rep)
        seen_trigrams.update(trigrams)
        running_tokens += rep_tokens

    return kept, gated, budget_reached


def resolve_auto_mode(*, model: str | None, compress: str) -> str:
    """Resolve ``compress="auto"`` to a concrete mode based on the query.

    Used by :func:`searcher.hybrid_search` to keep the auto-upgrade
    logic in one place. Fan-out queries default to ``"dedupe"``;
    single-model queries default to ``"none"`` for backward compat.
    Everything else passes through unchanged.
    """
    if compress != "auto":
        return compress
    return "dedupe" if model == "all" else "none"


# ── LLMLingua-2 adapter ───────────────────────────────────────────────


def _load_llmlingua_compressor():
    """Lazy-load the LLMLingua-2 compressor.

    Returns ``None`` and logs a warning if the optional extra isn't
    installed. Cached in the module-level ``_llmlingua_compressor``
    singleton so subsequent calls are a dict lookup.

    Broad ImportError catch: when the extra is missing, ``llmlingua``
    isn't on sys.path. When the extra is installed, ``llmlingua``
    itself can still fail to import on first use if its model-weight
    downloader hits a network error — we surface both as "mode=none"
    rather than crashing the search path.
    """
    global _llmlingua_compressor
    if _llmlingua_compressor is not None:
        return _llmlingua_compressor
    try:
        import logging

        from llmlingua import PromptCompressor

        logger = logging.getLogger("mempalace.compress")
        logger.info("Loading LLMLingua-2 compressor (first call)")
        # The ACL 2024 paper targets xlm-roberta-large — the default
        # weights for ``PromptCompressor`` with ``use_llmlingua2=True``.
        # Model weights land in ~/.cache/huggingface on first call.
        _llmlingua_compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
        )
        return _llmlingua_compressor
    except ImportError:
        return None
    except Exception as e:
        # Broad catch: any downstream failure (network, OOM, weight
        # corruption, tokenizer mismatch) must not break search. Log
        # and fall back to passthrough.
        import logging

        logging.getLogger("mempalace.compress").warning(
            "LLMLingua-2 load failed: %s — falling back to mode=none", e
        )
        return None


def _compress_llmlingua2(
    hits: list[dict],
    *,
    token_budget: int | None,
) -> tuple[list[dict], dict]:
    """Compress hits via LLMLingua-2's learned keep/drop classifier.

    Per-hit behavior:
      1. Compute the input token count via ``Dialect.count_tokens``.
      2. Derive a per-hit target: if ``token_budget`` is set, each hit
         gets ``max(1, token_budget // len(hits))`` output tokens so
         the total across the response stays under budget. Otherwise
         we let LLMLingua-2 pick its default compression ratio.
      3. Run the compressor on the drawer text; the result is a
         shortened string that preserves the most keep-labeled tokens.
      4. Rewrite the hit's ``text`` field with the compressed version,
         storing the original under ``original_text`` (publicly
         exposed in the response envelope so callers can recover the
         untrimmed drawer if they need to cite verbatim).

    Falls back to ``mode=none`` passthrough with a warning if the
    optional extra isn't installed or the compressor fails to load.
    The stats envelope still populates correctly so downstream
    callers see a consistent shape regardless of which path ran.
    """
    compressor = _load_llmlingua_compressor()
    input_tokens = sum(Dialect.count_tokens(h.get("text") or "") for h in hits)

    if compressor is None:
        # Fallback: passthrough + stats populated as if mode=none
        stats = _empty_stats("llmlingua2")
        stats["input_hits"] = len(hits)
        stats["output_hits"] = len(hits)
        stats["input_tokens"] = input_tokens
        stats["output_tokens"] = input_tokens
        stats["ratio"] = 1.0
        stats["fallback"] = "extras not installed"
        return list(hits), stats

    per_hit_budget = None
    if token_budget is not None and hits:
        per_hit_budget = max(1, token_budget // len(hits))

    out: list[dict] = []
    for hit in hits:
        text = hit.get("text") or ""
        if not text.strip():
            out.append(hit)
            continue

        new_hit = dict(hit)
        try:
            if per_hit_budget is not None:
                result = compressor.compress_prompt(
                    text,
                    target_token=per_hit_budget,
                    force_tokens=["\n"],
                )
            else:
                result = compressor.compress_prompt(text, force_tokens=["\n"])

            compressed = result.get("compressed_prompt", text) if isinstance(result, dict) else text
        except Exception as e:
            # Per-hit failure: keep the original, log, continue.
            import logging

            logging.getLogger("mempalace.compress").warning(
                "LLMLingua-2 compress_prompt failed on hit: %s", e
            )
            out.append(hit)
            continue

        new_hit["original_text"] = text
        new_hit["text"] = compressed
        new_hit["original_tokens"] = Dialect.count_tokens(text)
        new_hit["output_tokens"] = Dialect.count_tokens(compressed)
        out.append(new_hit)

    output_tokens = sum(
        h.get("output_tokens", Dialect.count_tokens(h.get("text") or "")) for h in out
    )
    stats = _empty_stats("llmlingua2")
    stats["input_hits"] = len(hits)
    stats["output_hits"] = len(out)
    stats["input_tokens"] = input_tokens
    stats["output_tokens"] = output_tokens
    stats["ratio"] = round(input_tokens / output_tokens, 3) if output_tokens else 0.0
    return out, stats


__all__ = [
    "MODES",
    "compress_results",
    "resolve_auto_mode",
]
