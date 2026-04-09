"""
embeddings.py — Configurable embedding model registry.

MemPalace supports multiple text embedding models coexisting in one palace.
Each model gets its own ChromaDB collection (dimensions can't be mixed
inside a single collection), identified by a stable ``slug``. The
registry declares available specs, advertises which optional ``pip``
extras are needed, and hands out lazy-loaded ``EmbeddingFunction``
instances on demand.

Design goals
------------

1. **Zero-cost import.** This module imports only stdlib at module
   scope. Heavy backends (``fastembed``, ``sentence_transformers``,
   ``ollama``) are imported inside ``load_embedding_function`` via
   ``importlib``. A user running ``mempalace status`` who never touches
   a non-default model pays nothing for the backends they don't have
   installed.

2. **Opportunistic weight load.** The ``EmbeddingFunction`` instance is
   constructed without touching model weights. The underlying
   ``SentenceTransformer(...)`` / ``TextEmbedding(...)`` / HTTP probe is
   deferred to the first ``__call__`` on the adapter. A 1 GB model
   weight file stays on disk until somebody actually runs a query
   against that model.

3. **Process-wide memoization.** ``_loaded_functions[slug]`` caches the
   constructed adapter so every call site in the same process (miner,
   searcher, MCP server, CLI) shares one instance and one loaded model.

4. **Stable slugs.** Slugs are what users type (``--model jina-code-v2``)
   and what config files persist (``default_embedding_model``). Model
   IDs (HuggingFace names, Ollama tags) are an implementation detail.

5. **Legacy-safe default.** ``slug="default"`` maps to Chroma's built-in
   ONNX all-MiniLM-L6-v2 via a **None passthrough**: the adapter
   returns ``None`` from ``load_embedding_function`` and the caller
   omits the ``embedding_function=`` kwarg when opening the collection.
   That preserves today's behavior byte-for-byte — existing palaces
   continue to work with no migration.

Collection naming
-----------------

``collection_name_for(slug)`` centralizes the naming scheme so callers
never string-interpolate:

* ``"default"`` → ``"mempalace_drawers"`` (the legacy name, unchanged)
* anything else → ``"mempalace_drawers__" + normalized_slug``

``normalized_slug`` replaces ``-`` and ``.`` with ``_`` so the
resulting collection name is a clean C identifier. The double
underscore prefix separates the namespace from legitimate wing/room
slugs (which use single dashes / underscores).
"""

import importlib
import importlib.util
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mempalace.embeddings")


# ── Slug ↔ collection name ────────────────────────────────────────────

_LEGACY_COLLECTION = "mempalace_drawers"
_COLLECTION_PREFIX = "mempalace_drawers__"
_SLUG_NORMALIZE_RE = re.compile(r"[^a-z0-9_]+")


def normalize_slug_for_collection(slug: str) -> str:
    """Make a slug safe to embed in a Chroma collection name.

    Lowercases, replaces any run of non-``[a-z0-9_]`` with a single
    underscore, strips leading/trailing underscores. Reversible only via
    the registry (we don't reconstruct the slug from the collection name).
    """
    return _SLUG_NORMALIZE_RE.sub("_", slug.lower()).strip("_")


def collection_name_for(slug: str) -> str:
    """Return the Chroma collection name bound to a model slug.

    The ``default`` slug maps to the legacy ``mempalace_drawers`` name so
    existing palaces continue to work without migration.
    """
    if slug == "default":
        return _LEGACY_COLLECTION
    return _COLLECTION_PREFIX + normalize_slug_for_collection(slug)


# ── Spec + registry ───────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbeddingSpec:
    """Static description of an embedding model.

    The ``factory`` is **not** stored on the spec — specs are frozen
    metadata. Factories are dispatched per-backend inside
    ``load_embedding_function``.

    Matryoshka Representation Learning (MRL) support: models trained
    with MRL produce embeddings where the first N dimensions are
    independently usable with minimal recall loss. Set
    ``supports_matryoshka=True`` on the spec so callers know truncation
    is safe; set ``truncate_dim`` to actually slice every returned
    vector to that many dimensions before it reaches Chroma. Truncation
    happens inside the adapter's ``__call__`` so Chroma stores and
    compares lower-dim vectors — yielding proportional storage
    reduction and HNSW speedup.
    """

    slug: str
    display_name: str
    description: str
    backend: str  # "chroma-default" | "fastembed" | "sentence-transformers" | "ollama"
    model_id: str
    dimension: int
    context_tokens: int
    extras_required: tuple[str, ...] = field(default_factory=tuple)
    supports_matryoshka: bool = False
    truncate_dim: int | None = None


REGISTRY: dict[str, EmbeddingSpec] = {}


def register(spec: EmbeddingSpec) -> None:
    """Add a spec to the process-wide registry. Later entries win."""
    REGISTRY[spec.slug] = spec


def list_specs() -> list[EmbeddingSpec]:
    """Return every registered spec in registration order."""
    return list(REGISTRY.values())


def get_spec(slug: str) -> EmbeddingSpec:
    """Look up a spec by slug. Raises ``KeyError`` on unknown slug."""
    try:
        return REGISTRY[slug]
    except KeyError as e:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Unknown embedding model {slug!r}. Available: {available}") from e


def is_installed(spec: EmbeddingSpec) -> bool:
    """Check whether the spec's extras are importable without actually
    importing them.

    Uses ``importlib.util.find_spec`` which probes ``sys.path`` but
    never triggers module import or weight download. Zero-cost.
    """
    return all(importlib.util.find_spec(extra) is not None for extra in spec.extras_required)


# ── Opportunistic loading ─────────────────────────────────────────────

# Module-import-scope cache of constructed embedding functions. Keyed on
# slug so every caller in the same process shares one instance — and
# hence one loaded model. Populated lazily on first
# ``load_embedding_function`` call for a given slug.
_loaded_functions: dict[str, Any] = {}


class _MissingExtrasError(RuntimeError):
    """Raised when a backend's optional extras aren't installed."""


def _require_extras(spec: EmbeddingSpec) -> None:
    if not is_installed(spec):
        extras = ", ".join(spec.extras_required)
        extra_name = "embeddings-" + spec.backend.replace("chroma-default", "default").replace(
            "sentence-transformers", "sentence-transformers"
        )
        raise _MissingExtrasError(
            f"Model {spec.slug!r} requires the {extras!r} package(s). "
            f"Install with:  pip install 'mempalace[{extra_name}]'"
        )


def load_embedding_function(slug: str) -> Any:
    """Return a Chroma-compatible embedding function for the given slug.

    **Opportunistic load gate.** The backend module is imported only
    here. The underlying model weights are loaded even later, on the
    first ``__call__`` of the returned adapter.

    Returns ``None`` for the ``default`` slug — the caller should omit
    ``embedding_function=`` when opening the collection, letting Chroma
    use its built-in ONNX mini-lm.

    Cached: repeated calls with the same slug return the same instance.
    """
    if slug in _loaded_functions:
        return _loaded_functions[slug]

    spec = get_spec(slug)

    if spec.backend == "chroma-default":
        # Sentinel: the caller should omit embedding_function= entirely
        # so Chroma uses its internal default. We cache ``None`` to make
        # the hit path a single dict lookup.
        _loaded_functions[slug] = None
        return None

    _require_extras(spec)

    if spec.backend == "fastembed":
        fn = _FastEmbedFn(spec)
    elif spec.backend == "sentence-transformers":
        fn = _SentenceTransformerFn(spec)
    elif spec.backend == "ollama":
        fn = _OllamaFn(spec)
    else:
        raise ValueError(f"Unknown backend {spec.backend!r} for model {slug!r}")

    _loaded_functions[slug] = fn
    return fn


def clear_cache() -> None:
    """Drop every cached embedding function. Useful for tests that swap
    specs in and out of the registry between runs.
    """
    _loaded_functions.clear()


# ── Backend adapters ──────────────────────────────────────────────────
#
# Each adapter implements Chroma's EmbeddingFunction[Documents] protocol:
#   __call__(self, input: list[str]) -> list[list[float]]
# The underlying model is held in a private attribute and loaded lazily
# on first __call__.


def _apply_matryoshka(vectors: list[list[float]], spec: EmbeddingSpec) -> list[list[float]]:
    """Slice vectors to spec.truncate_dim if set.

    Guards against specs that set ``truncate_dim`` without
    ``supports_matryoshka=True`` — truncating a non-MRL embedding
    corrupts it, so we refuse rather than silently producing
    garbage vectors.

    The truncation ratio is logged once per load so users see what
    they're getting. Callers never observe the full-length vector
    when truncation is active.
    """
    if spec.truncate_dim is None:
        return vectors
    if not spec.supports_matryoshka:
        raise ValueError(
            f"Model {spec.slug!r} has truncate_dim={spec.truncate_dim} but "
            f"supports_matryoshka=False. Truncating a non-MRL embedding "
            f"corrupts the vector — set supports_matryoshka=True on the spec "
            f"only if the model was trained with Matryoshka Representation "
            f"Learning."
        )
    dim = spec.truncate_dim
    return [vec[:dim] for vec in vectors]


class _FastEmbedFn:
    """fastembed-backed embedding function.

    fastembed is a pure-ONNX library (no torch) with a curated catalog
    of popular small/medium embedding models. Model weights are cached
    in ``~/.cache/fastembed/`` by default and pulled over HTTP on first
    use.
    """

    def __init__(self, spec: EmbeddingSpec):
        self._spec = spec
        self._model: Any | None = None

    def _ensure(self):
        if self._model is None:
            fastembed = importlib.import_module("fastembed")
            logger.info("Loading fastembed model %s", self._spec.model_id)
            self._model = fastembed.TextEmbedding(model_name=self._spec.model_id)
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = self._ensure()
        # fastembed returns a generator of numpy arrays; materialize.
        vectors = [vec.tolist() for vec in model.embed(list(input))]
        return _apply_matryoshka(vectors, self._spec)

    # Chroma looks at __class__.__name__ for identity; give it something
    # unique per spec so two different specs don't collide in Chroma's
    # internal caches.
    def name(self) -> str:
        return f"fastembed:{self._spec.slug}"


class _SentenceTransformerFn:
    """sentence-transformers-backed embedding function.

    Full HuggingFace catalog available. Drags torch transitively —
    heavier install but covers SOTA models fastembed doesn't carry.
    """

    def __init__(self, spec: EmbeddingSpec):
        self._spec = spec
        self._model: Any | None = None

    def _ensure(self):
        if self._model is None:
            st = importlib.import_module("sentence_transformers")
            logger.info("Loading sentence-transformers model %s", self._spec.model_id)
            self._model = st.SentenceTransformer(self._spec.model_id)
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = self._ensure()
        vectors = model.encode(list(input), convert_to_numpy=True)
        materialized = [vec.tolist() for vec in vectors]
        return _apply_matryoshka(materialized, self._spec)

    def name(self) -> str:
        return f"st:{self._spec.slug}"


class _OllamaFn:
    """Ollama HTTP-backed embedding function.

    Calls the local Ollama server's ``/api/embeddings`` endpoint. No
    model weights are managed by MemPalace — the user runs
    ``ollama pull <model>`` separately. Zero extra download from our
    side.
    """

    def __init__(self, spec: EmbeddingSpec):
        self._spec = spec
        self._client: Any | None = None

    def _ensure(self):
        if self._client is None:
            try:
                ollama = importlib.import_module("ollama")
            except ImportError as e:
                raise _MissingExtrasError(
                    f"Model '{self._spec.slug}' requires the ollama Python client. "
                    "Install with:  pip install 'mempalace[embeddings-ollama]'"
                ) from e
            self._client = ollama.Client()
        return self._client

    def __call__(self, input: list[str]) -> list[list[float]]:
        client = self._ensure()
        out: list[list[float]] = []
        for text in input:
            try:
                resp = client.embeddings(model=self._spec.model_id, prompt=text)
            except Exception as e:
                # Broad catch: the ollama client wraps httpx and can raise
                # any of ConnectionError, httpx.HTTPError, RuntimeError, or
                # its own ResponseError subclasses — we translate connection
                # failures into a friendlier RuntimeError and re-raise the rest.
                msg = str(e)
                if "connection" in msg.lower() or "refused" in msg.lower():
                    raise RuntimeError(
                        "Ollama server unreachable at http://localhost:11434 "
                        "— start it with `ollama serve` and confirm the "
                        f"{self._spec.model_id!r} model has been pulled "
                        f"with `ollama pull {self._spec.model_id}`."
                    ) from e
                raise
            out.append(list(resp["embedding"]))
        return _apply_matryoshka(out, self._spec)

    def name(self) -> str:
        return f"ollama:{self._spec.slug}"


# ── Built-in registry seed (v1) ───────────────────────────────────────


def _register_builtins() -> None:
    """Populate the registry with the seven v1 specs.

    Called once at module import. Idempotent — re-calling rebuilds the
    same entries.
    """

    register(
        EmbeddingSpec(
            slug="default",
            display_name="Chroma default (all-MiniLM-L6-v2)",
            description=(
                "ONNX, 384-dim, 256 tokens. Built into ChromaDB. Good "
                "for short English prose. No extras required."
            ),
            backend="chroma-default",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            context_tokens=256,
            extras_required=(),
        )
    )
    register(
        EmbeddingSpec(
            slug="bge-small-en",
            display_name="BGE Small EN v1.5",
            description=(
                "Small general retrieval model via fastembed. 384-dim, "
                "512-token context. Faster and cheaper than the default "
                "with marginally better retrieval."
            ),
            backend="fastembed",
            model_id="BAAI/bge-small-en-v1.5",
            dimension=384,
            context_tokens=512,
            extras_required=("fastembed",),
        )
    )
    register(
        EmbeddingSpec(
            slug="nomic-text-v1.5",
            display_name="Nomic Embed Text v1.5",
            description=(
                "768-dim, 8192-token context. Ideal for long LLM "
                "conversations, decision logs, and architecture docs — "
                "the 8k window means no mid-document truncation. "
                "Matryoshka-trained: truncate to 256 or 128 dims for "
                "3-6× storage reduction with minimal recall loss."
            ),
            backend="fastembed",
            model_id="nomic-ai/nomic-embed-text-v1.5",
            dimension=768,
            context_tokens=8192,
            extras_required=("fastembed",),
            supports_matryoshka=True,
        )
    )
    register(
        EmbeddingSpec(
            slug="jina-code-v2",
            display_name="Jina Embeddings v2 Base Code",
            description=(
                "768-dim, 8192-token context. Trained on CodeSearchNet — "
                "strongest small local model for source code retrieval."
            ),
            backend="fastembed",
            model_id="jinaai/jina-embeddings-v2-base-code",
            dimension=768,
            context_tokens=8192,
            extras_required=("fastembed",),
        )
    )
    register(
        EmbeddingSpec(
            slug="mxbai-large",
            display_name="MixedBread MXBAI Embed Large v1",
            description=(
                "1024-dim, 512-token context. MTEB top-5 general "
                "retrieval. Drags torch via sentence-transformers. "
                "Matryoshka-trained: truncate to 512 or 256 dims for "
                "meaningful storage reduction with minimal recall loss."
            ),
            backend="sentence-transformers",
            model_id="mixedbread-ai/mxbai-embed-large-v1",
            dimension=1024,
            context_tokens=512,
            extras_required=("sentence_transformers",),
            supports_matryoshka=True,
        )
    )
    register(
        EmbeddingSpec(
            slug="bge-m3",
            display_name="BGE-M3 (multilingual + long context)",
            description=(
                "1024-dim, 8192-token context. BAAI's unified model "
                "producing dense + sparse + multi-vector from one "
                "encoder (dense-only mode used here). Multilingual "
                "across 100+ languages. Strong on MIRACL and MLDR. "
                "Matryoshka-compatible: truncate to 512 or 256 dims "
                "for storage reduction."
            ),
            backend="fastembed",
            model_id="BAAI/bge-m3",
            dimension=1024,
            context_tokens=8192,
            extras_required=("fastembed",),
            supports_matryoshka=True,
        )
    )
    register(
        EmbeddingSpec(
            slug="ollama-nomic",
            display_name="Ollama: nomic-embed-text",
            description=(
                "Uses the locally-running Ollama server. Zero extra "
                "downloads if you've already run `ollama pull "
                "nomic-embed-text`."
            ),
            backend="ollama",
            model_id="nomic-embed-text",
            dimension=768,
            context_tokens=8192,
            extras_required=("ollama",),
        )
    )
    register(
        EmbeddingSpec(
            slug="ollama-mxbai",
            display_name="Ollama: mxbai-embed-large",
            description=(
                "Same Ollama path with mxbai-embed-large as the backing "
                "model — 1024-dim retrieval quality without pulling "
                "torch or fastembed."
            ),
            backend="ollama",
            model_id="mxbai-embed-large",
            dimension=1024,
            context_tokens=512,
            extras_required=("ollama",),
        )
    )


_register_builtins()
