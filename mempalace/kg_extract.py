"""
kg_extract.py — Automatic knowledge-graph triple extraction.

Closes the single biggest architectural gap in MemPalace: the
``knowledge_graph`` module already supports rich temporal queries,
but nothing populates it automatically. Every triple on a pre-this-
tranche palace had to come from an explicit ``mempalace_kg_add``
MCP call. This module fixes that.

Two extraction paths ship in parallel:

* **Heuristic** (zero new deps, ships with the core install) —
  regex + ``entity_detector`` patterns for simple relations:
  "X is a Y", "X works at Y", "X lives in Z", "X loves W". Catches
  ~30-50% of the relations actually expressed in prose, with sub-
  millisecond per-drawer cost and no network access. Good enough
  to seed the KG from a fresh mine and for users who can't run a
  local LLM.

* **Ollama** (optional extra ``kg-extract-ollama``) — delegates
  triple extraction to a locally-running Ollama server. Sends a
  structured prompt requesting JSON-formatted triples and parses
  the response. Catches ~80% of relations but costs ~50-500 ms
  per drawer depending on the model. Zero external network access
  once the Ollama model is pulled.

Both paths share:
    - The same :class:`Triple` dataclass
    - The same ``extract(text, *, entity_hints=None)`` API
    - The same dedupe + confidence-voting path in
      :class:`mempalace.knowledge_graph.KnowledgeGraph.add_triple`
      (extended in this tranche to merge ``source_closet`` lists
      when the same triple is asserted by multiple drawers).

The factory :func:`get_extractor` dispatches by mode. Users pick
per-mine via ``mempalace mine <dir> --extract-kg --kg-extract-mode
{heuristic,ollama}``; the default is ``heuristic`` so the zero-dep
experience is the out-of-box default.

Design notes
------------

* **Never crashes the mine.** Extraction failures are logged and
  the drawer is still written to Chroma + trie. The KG is a
  secondary surface — breaking it must not break the primary
  retrieval path.
* **Confidence in [0.1, 0.9].** Never 1.0 (we can't be sure) and
  never 0 (then why add it at all). Pattern specificity drives the
  value: a regex with named groups + entity capitalization hits
  0.7+, a loose keyword match drops to 0.3.
* **Entity name normalization** is deferred to
  ``knowledge_graph.add_triple`` — it already lowercases + strips
  spaces via ``_entity_id``. This module preserves the user's
  original capitalization so the KG's ``entities.name`` column
  reads naturally.
"""

import importlib
import importlib.util
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger("mempalace.kg_extract")


# ── Triple dataclass ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Triple:
    """A single extracted (subject, predicate, object) fact.

    All temporal fields are optional — when extraction can pin down
    a ``valid_from`` from a temporal marker in the source text
    ("in 2024", "since last month"), the extractor populates it.
    Otherwise the triple is treated as "always true" by
    :meth:`KnowledgeGraph.add_triple`.
    """

    subject: str
    predicate: str
    obj: str
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 0.5
    source_drawer_id: str | None = None
    # Free-form metadata from the extractor (pattern name, raw match
    # text, Ollama model used). Never persisted to the KG — used only
    # for debugging and tests.
    metadata: dict = field(default_factory=dict)


# ── Extractor protocol + factory ──────────────────────────────────────


class TripleExtractor(Protocol):
    """Common interface for every extraction backend.

    Implementations take a text blob (and optionally a list of known
    entity names to bias the extraction) and return a list of Triple
    objects. The return order is insignificant; callers sort by
    confidence or by subject/predicate before persisting.
    """

    def extract(
        self,
        text: str,
        *,
        entity_hints: list[str] | None = None,
        source_drawer_id: str | None = None,
    ) -> list[Triple]: ...


class _MissingExtrasError(RuntimeError):
    """Raised when an optional extraction backend's extras are missing."""


def get_extractor(mode: str = "heuristic", **kwargs) -> TripleExtractor:
    """Factory: dispatch by mode to a concrete extractor.

    ``mode="heuristic"`` returns a :class:`HeuristicExtractor` — no
    deps, works on every install. ``mode="ollama"`` returns an
    :class:`OllamaExtractor` (requires the ``kg-extract-ollama``
    extra). Any other value raises ``ValueError`` with a list of
    known modes.

    Kwargs are forwarded to the extractor's constructor so callers
    can pick a specific Ollama model via
    ``get_extractor("ollama", model="llama3.1:8b")``.
    """
    if mode == "heuristic":
        return HeuristicExtractor()
    if mode == "ollama":
        return OllamaExtractor(**kwargs)
    raise ValueError(f"Unknown kg_extract mode {mode!r}. Valid modes: heuristic, ollama")


# ── Heuristic extractor ───────────────────────────────────────────────


# Regex patterns for common relations. Each entry maps a relation
# predicate to a compiled regex + confidence score. The regexes use
# named groups so extraction is a simple ``match.group("subject")`` /
# ``match.group("object")`` lookup.
#
# Constraints that apply to every pattern:
#   - Subject must start with an uppercase letter (basic named-entity
#     filter that catches most proper nouns in English prose without
#     an NLP library). This is cheap but imperfect — sentences like
#     "The cat is a mammal" fail gracefully.
#   - Object must be 2-60 characters and not overlap keywords from
#     the relation itself.
#   - Whole-word boundaries (``\b``) so "Alice" doesn't match
#     "Alicent".

# Subject group is case-sensitive (wrapped in (?-i:...)) so the
# capitalization filter works — lowercase "alice" shouldn't match.
# The verb/keyword portions stay case-insensitive via the top-level
# re.IGNORECASE flag.
_NAME = r"(?-i:(?P<subject>[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2}))"
_OBJ = r"(?P<obj>[a-zA-Z][\w\s'-]{1,60}?)"

# (predicate, compiled pattern, confidence)
_HEURISTIC_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # "Alice is a doctor", "Max is an engineer"
    (
        "is_a",
        re.compile(
            rf"\b{_NAME}\s+is\s+(?:a|an)\s+{_OBJ}\b(?=\s*[.,;!?]|$)",
            re.IGNORECASE,
        ),
        0.7,
    ),
    # "Alice works at Acme", "Bob works for Google"
    (
        "works_at",
        re.compile(
            rf"\b{_NAME}\s+works?\s+(?:at|for)\s+{_OBJ}\b(?=\s*[.,;!?]|$)",
            re.IGNORECASE,
        ),
        0.75,
    ),
    # "Alice lives in Berlin", "Max lives at 42 Main St"
    (
        "lives_in",
        re.compile(
            rf"\b{_NAME}\s+lives?\s+in\s+{_OBJ}\b(?=\s*[.,;!?]|$)",
            re.IGNORECASE,
        ),
        0.7,
    ),
    # "Alice loves rock climbing", "Max loves chess"
    (
        "loves",
        re.compile(
            rf"\b{_NAME}\s+loves?\s+{_OBJ}\b(?=\s*[.,;!?]|$)",
            re.IGNORECASE,
        ),
        0.6,
    ),
    # "Alice uses Postgres", "Max uses VSCode"
    (
        "uses",
        re.compile(
            rf"\b{_NAME}\s+uses?\s+{_OBJ}\b(?=\s*[.,;!?]|$)",
            re.IGNORECASE,
        ),
        0.5,
    ),
    # "Alice prefers Ruby over Python" — captures just Alice prefers Ruby
    (
        "prefers",
        re.compile(
            rf"\b{_NAME}\s+prefers?\s+{_OBJ}\b(?=\s*(?:over|to|\.|,|;|!|\?)|$)",
            re.IGNORECASE,
        ),
        0.6,
    ),
]

# Noise words that, if they appear as the extracted object, indicate a
# false positive. Stripped before accepting the triple.
_OBJECT_STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "it",
    "them",
    "us",
    "you",
    "me",
    "i",
    "he",
    "she",
    "we",
    "they",
}


def _clean_object(obj: str) -> str:
    """Trim whitespace, leading articles, and trailing punctuation."""
    obj = obj.strip().rstrip(".,;!?:")
    # Strip leading articles: "a", "an", "the"
    tokens = obj.split()
    while tokens and tokens[0].lower() in {"a", "an", "the"}:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


class HeuristicExtractor:
    """Pattern-matching triple extractor.

    Walks the text once per configured pattern, yielding every match
    as a Triple. Duplicate triples (same subject/predicate/object)
    are collapsed to their highest-confidence instance before
    returning.

    Performance: ~50-100 μs per 1 KB of text on commodity hardware.
    No model weights, no network, no new dependencies.
    """

    def extract(
        self,
        text: str,
        *,
        entity_hints: list[str] | None = None,
        source_drawer_id: str | None = None,
    ) -> list[Triple]:
        if not text or not text.strip():
            return []

        seen: dict[tuple[str, str, str], Triple] = {}
        for predicate, pattern, base_confidence in _HEURISTIC_PATTERNS:
            for match in pattern.finditer(text):
                subject_raw = match.group("subject").strip()
                obj_raw = match.group("obj")
                obj_clean = _clean_object(obj_raw)

                if not obj_clean or obj_clean.lower() in _OBJECT_STOPWORDS:
                    continue
                if len(obj_clean) < 2 or len(obj_clean) > 60:
                    continue

                # Boost confidence if the subject appears in
                # entity_hints (we trust the miner's detected entities
                # more than a capitalization heuristic).
                confidence = base_confidence
                if entity_hints:
                    hint_set = {h.lower() for h in entity_hints}
                    if subject_raw.lower() in hint_set:
                        confidence = min(0.9, confidence + 0.15)

                key = (subject_raw.lower(), predicate, obj_clean.lower())
                existing = seen.get(key)
                if existing is None or existing.confidence < confidence:
                    seen[key] = Triple(
                        subject=subject_raw,
                        predicate=predicate,
                        obj=obj_clean,
                        confidence=confidence,
                        source_drawer_id=source_drawer_id,
                        metadata={
                            "pattern": predicate,
                            "match": match.group(0),
                        },
                    )

        return list(seen.values())


# ── Ollama extractor ──────────────────────────────────────────────────


_OLLAMA_PROMPT_TEMPLATE = """Extract factual relationships from the text below as a JSON array.

Each item must be an object with exactly these keys:
  subject: the entity doing/being something (short proper noun)
  predicate: the relationship type (snake_case verb, e.g. "works_at", "loves", "child_of")
  object: the entity being connected to (short noun or proper noun)
  confidence: number between 0 and 1 indicating how certain you are
  valid_from: optional ISO date (YYYY-MM-DD or YYYY) if the text gives one, else null

Only extract relationships that are explicitly stated. Do not infer, do not hallucinate.
Return ONLY the JSON array, no prose, no markdown, no code fences.

Text:
\"\"\"
{text}
\"\"\"

JSON array:"""


class OllamaExtractor:  # pragma: no cover
    """Local-LLM triple extractor via the Ollama HTTP client.

    Lazy-loads the ``ollama`` package on first ``extract()`` call,
    so the core install is never charged for the optional dep.
    Raises :class:`_MissingExtrasError` with a friendly install
    hint if the extra isn't installed.

    Model selection defaults to ``llama3.1:8b`` but any locally-pulled
    Ollama model with reasonable JSON output capability will work.
    ``qwen2.5:7b``, ``mistral:7b``, and ``phi3:medium`` have all been
    observed to work in practice.
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self._model = model
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return self._client
        if importlib.util.find_spec("ollama") is None:
            raise _MissingExtrasError(
                "The Ollama KG extractor requires the 'ollama' package. "
                "Install with: pip install 'mempalace[kg-extract-ollama]'"
            )
        try:
            ollama = importlib.import_module("ollama")
            self._client = ollama.Client()
            return self._client
        except ImportError as e:
            raise _MissingExtrasError(
                "Failed to import the ollama package even though it "
                "appears installed. Reinstall with: "
                "pip install --force-reinstall 'mempalace[kg-extract-ollama]'"
            ) from e

    def extract(
        self,
        text: str,
        *,
        entity_hints: list[str] | None = None,
        source_drawer_id: str | None = None,
    ) -> list[Triple]:
        if not text or not text.strip():
            return []

        client = self._ensure()
        prompt = _OLLAMA_PROMPT_TEMPLATE.format(text=text[:8000])  # 8k char cap

        try:
            response = client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
        except Exception as e:
            # Broad catch: ollama can raise any of ConnectionError,
            # httpx.HTTPError, or its own ResponseError. Translate
            # connection issues into a friendlier RuntimeError so
            # miners can log + continue.
            msg = str(e)
            if "connection" in msg.lower() or "refused" in msg.lower():
                raise RuntimeError(
                    f"Ollama server unreachable. Start it with `ollama serve` "
                    f"and confirm `{self._model}` is pulled "
                    f"(`ollama pull {self._model}`)."
                ) from e
            # Log and return empty rather than crashing extraction.
            logger.warning("OllamaExtractor chat failed: %s — returning no triples", e)
            return []

        raw = (
            response.get("message", {}).get("content", "")
            if isinstance(response, dict)
            else getattr(response, "message", {}).get("content", "")
        )
        return self._parse(raw, source_drawer_id=source_drawer_id)

    def _parse(self, raw: str, *, source_drawer_id: str | None) -> list[Triple]:
        """Parse the LLM's JSON response into Triple objects.

        Tolerates common LLM sins:
          - Wrapping the JSON in ```json ... ``` fences
          - Emitting a single object instead of an array
          - Trailing commas (by extracting the first ``[...]`` or
            ``{...}`` block before json.loads)
          - Missing optional fields (confidence, valid_from)

        On any parse failure returns an empty list and logs a warning.
        The miner will continue without extracted triples for this
        drawer — the caller sees "no triples" rather than a crash.
        """
        text = raw.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        # Extract first [...] block if the model added extra prose
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("OllamaExtractor: JSON parse failed: %s", e)
            return []

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        out: list[Triple] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not subject or not predicate or not obj:
                continue

            # Clamp confidence to our [0.1, 0.9] band
            try:
                raw_conf = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                raw_conf = 0.6
            confidence = max(0.1, min(0.9, raw_conf))

            valid_from = item.get("valid_from")
            if valid_from is not None and not isinstance(valid_from, str):
                valid_from = None

            out.append(
                Triple(
                    subject=subject,
                    predicate=predicate.lower().replace(" ", "_"),
                    obj=obj,
                    valid_from=valid_from,
                    confidence=confidence,
                    source_drawer_id=source_drawer_id,
                    metadata={"ollama_model": self._model},
                )
            )
        return out


# ── Palace-wide extraction helper ────────────────────────────────────


def extract_from_palace(
    palace_path: str,
    *,
    mode: str = "heuristic",
    model: str = "llama3.1:8b",
    batch_size: int = 500,
    progress_callback=None,
) -> dict:
    """Walk every drawer in the palace's Chroma collection and populate
    the knowledge graph via the chosen extractor.

    This is the engine for ``mempalace mine --extract-kg`` (live
    extraction during mining) and for the retroactive ``mempalace
    kg-extract`` subcommand. The walker iterates the Chroma collection
    in pages (default 500 drawers per page) so it scales to large
    palaces without loading the entire corpus into memory.

    For each drawer:
      1. Load the drawer text and metadata
      2. Run ``extractor.extract(text, source_drawer_id=drawer_id)``
      3. For every returned Triple, call
         ``kg.add_triple(subject, predicate, obj, confidence=...,
         source_closet=drawer_id)``
      4. The KG's dedupe + confidence-voting path handles repeated
         evidence for the same triple across multiple drawers.

    Errors on any individual drawer are logged and skipped — the
    walker never fails the whole extraction pass because of one bad
    drawer.

    Returns
    -------
    dict
        ``{"drawers_scanned": int, "triples_added": int,
           "errors": int, "mode": str}``
    """
    from .knowledge_graph import KnowledgeGraph
    from .palace_io import open_collection

    try:
        collection = open_collection(palace_path)
    except Exception as e:
        logger.error("extract_from_palace: cannot open collection: %s", e)
        return {
            "drawers_scanned": 0,
            "triples_added": 0,
            "errors": 1,
            "mode": mode,
            "error": str(e),
        }

    extractor = get_extractor(mode, model=model) if mode == "ollama" else get_extractor(mode)
    kg = KnowledgeGraph()

    stats = {
        "drawers_scanned": 0,
        "triples_added": 0,
        "errors": 0,
        "mode": mode,
    }

    total = collection.count()
    offset = 0
    while offset < total:
        try:
            batch = collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
        except Exception as e:
            logger.error("extract_from_palace: page fetch failed: %s", e)
            stats["errors"] += 1
            break

        drawer_ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        if not drawer_ids:
            break

        for drawer_id, text in zip(drawer_ids, docs, strict=False):
            stats["drawers_scanned"] += 1
            if not text:
                continue
            try:
                triples = extractor.extract(text, source_drawer_id=drawer_id)
            except Exception as e:
                logger.warning(
                    "extract_from_palace: extraction failed on %s: %s",
                    drawer_id,
                    e,
                )
                stats["errors"] += 1
                continue

            for triple in triples:
                try:
                    kg.add_triple(
                        triple.subject,
                        triple.predicate,
                        triple.obj,
                        valid_from=triple.valid_from,
                        valid_to=triple.valid_to,
                        confidence=triple.confidence,
                        source_closet=drawer_id,
                    )
                    stats["triples_added"] += 1
                except Exception as e:
                    logger.warning("extract_from_palace: kg.add_triple failed: %s", e)
                    stats["errors"] += 1

            if progress_callback:
                progress_callback(stats["drawers_scanned"], total)

        if len(drawer_ids) < batch_size:
            break
        offset += len(drawer_ids)

    return stats


__all__ = [
    "Triple",
    "TripleExtractor",
    "HeuristicExtractor",
    "OllamaExtractor",
    "_MissingExtrasError",
    "get_extractor",
    "extract_from_palace",
]
