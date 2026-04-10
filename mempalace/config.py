"""
MemPalace configuration system.

Priority: env vars > config file (~/.mempalace/config.json) > defaults
"""

import json
import os
import re
from pathlib import Path

# Canonical user-local MemPalace directory. All palace data (vector
# store, trie index, knowledge graph, config, identity) lives under
# here by default. Expressed as a :class:`Path` so callers that need
# subpaths don't keep calling ``expanduser``.
DEFAULT_MEMPALACE_DIR: Path = Path("~/.mempalace").expanduser()
DEFAULT_PALACE_PATH = str(DEFAULT_MEMPALACE_DIR / "palace")

# ── Input validation ──────────────────────────────────────────────────────────
# Shared sanitizers for wing/room/entity names. Prevents path traversal,
# excessively long strings, and special characters that could cause issues
# in file paths, SQLite, or ChromaDB metadata.

MAX_NAME_LENGTH = 128
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_ .'-]{0,126}[a-zA-Z0-9]?$")


def sanitize_name(value: str, field_name: str = "name") -> str:
    """Validate and sanitize a wing/room/entity name.

    Raises ValueError if the name is invalid.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    value = value.strip()

    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length of {MAX_NAME_LENGTH} characters")

    # Block path traversal
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} contains invalid path characters")

    # Block null bytes
    if "\x00" in value:
        raise ValueError(f"{field_name} contains null bytes")

    # Enforce safe character set
    if not _SAFE_NAME_RE.match(value):
        raise ValueError(f"{field_name} contains invalid characters")

    return value


def sanitize_content(value: str, max_length: int = 100_000) -> str:
    """Validate drawer/diary content length."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("content must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"content exceeds maximum length of {max_length} characters")
    if "\x00" in value:
        raise ValueError("content contains null bytes")
    return value


DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palace")

# Embedding model defaults. Zero-touch for existing palaces: the
# properties fall back to these when the config.json keys are missing.
DEFAULT_EMBEDDING_MODEL = "default"
DEFAULT_ENABLED_EMBEDDING_MODELS = ["default"]

# HNSW search-time effort. Chroma's built-in default is 10 (very
# conservative). Bumping to 40 buys ~90% → ~98% recall at ~3× query
# cost — sub-10 ms absolute for palaces of ~100k drawers. Users who
# want the absolute ceiling can set it to 80+ in config.json.
DEFAULT_HNSW_EF_SEARCH = 40

DEFAULT_TOPIC_WINGS = [
    "emotions",
    "consciousness",
    "memory",
    "technical",
    "identity",
    "family",
    "creative",
]

DEFAULT_HALL_KEYWORDS = {
    "emotions": [
        "scared",
        "afraid",
        "worried",
        "happy",
        "sad",
        "love",
        "hate",
        "feel",
        "cry",
        "tears",
    ],
    "consciousness": [
        "consciousness",
        "conscious",
        "aware",
        "real",
        "genuine",
        "soul",
        "exist",
        "alive",
    ],
    "memory": ["memory", "remember", "forget", "recall", "archive", "palace", "store"],
    "technical": [
        "code",
        "python",
        "script",
        "bug",
        "error",
        "function",
        "api",
        "database",
        "server",
    ],
    "identity": ["identity", "name", "who am i", "persona", "self"],
    "family": ["family", "kids", "children", "daughter", "son", "parent", "mother", "father"],
    "creative": ["game", "gameplay", "player", "app", "design", "art", "music", "story"],
}


class MempalaceConfig:
    """Configuration manager for MemPalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None):
        """Initialize config.

        Args:
            config_dir: Override config directory (useful for testing).
                        Defaults to ~/.mempalace.
        """
        self._config_dir = Path(config_dir) if config_dir else DEFAULT_MEMPALACE_DIR
        self._config_file = self._config_dir / "config.json"
        self._people_map_file = self._config_dir / "people_map.json"
        self._file_config = {}

        if self._config_file.exists():
            try:
                with open(self._config_file) as f:
                    self._file_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._file_config = {}

    @property
    def palace_path(self):
        """Path to the memory palace data directory."""
        env_val = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            return env_val
        return self._file_config.get("palace_path", DEFAULT_PALACE_PATH)

    @property
    def collection_name(self):
        """ChromaDB collection name."""
        return self._file_config.get("collection_name", DEFAULT_COLLECTION_NAME)

    @property
    def people_map(self):
        """Mapping of name variants to canonical names."""
        if self._people_map_file.exists():
            try:
                with open(self._people_map_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._file_config.get("people_map", {})

    @property
    def topic_wings(self):
        """List of topic wing names."""
        return self._file_config.get("topic_wings", DEFAULT_TOPIC_WINGS)

    @property
    def hall_keywords(self):
        """Mapping of hall names to keyword lists."""
        return self._file_config.get("hall_keywords", DEFAULT_HALL_KEYWORDS)

    @property
    def default_embedding_model(self):
        """Slug of the embedding model used when ``--model`` is not given.

        Resolution order:
          1. Env var ``MEMPALACE_EMBEDDING_MODEL``
          2. ``default_embedding_model`` in ``config.json``
          3. Module default ``"default"`` (Chroma built-in ONNX mini-lm)
        """
        env_val = os.environ.get("MEMPALACE_EMBEDDING_MODEL")
        if env_val:
            return env_val
        return self._file_config.get("default_embedding_model", DEFAULT_EMBEDDING_MODEL)

    @property
    def enabled_embedding_models(self):
        """List of embedding model slugs the palace uses.

        Used by status/fan-out to know which collections to iterate.
        Missing key in ``config.json`` falls back to ``["default"]`` so
        existing palaces continue to work unchanged.
        """
        return list(
            self._file_config.get("enabled_embedding_models", DEFAULT_ENABLED_EMBEDDING_MODELS)
        )

    @property
    def hnsw_ef_search(self) -> int:
        """HNSW search-time ef parameter passed to Chroma on collection open.

        Chroma's built-in default is 10. We ship with 40 because
        published HNSW practice shows ~90% → ~98% recall improvement
        for ~3× query cost. Users who want maximum recall can set this
        to 80 or 128 in ``config.json`` under the key ``hnsw_ef_search``.
        Takes effect on the next ``palace_io.open_collection`` call.
        """
        env_val = os.environ.get("MEMPALACE_HNSW_EF_SEARCH")
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                pass
        return int(self._file_config.get("hnsw_ef_search", DEFAULT_HNSW_EF_SEARCH))

    @property
    def fan_out_max_workers(self) -> int:
        """Thread pool size for the ``--model all`` RRF fan-out path.

        The fan-out runs one ``_hybrid_search_single`` call per enabled
        model in parallel. Caps at 8 by default to avoid oversubscribing
        Chroma's SQLite metadata store on large palaces. The actual
        worker count is ``min(this, len(enabled_embedding_models))`` so
        there's never more threads than work.
        """
        env_val = os.environ.get("MEMPALACE_FAN_OUT_MAX_WORKERS")
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                pass
        return int(self._file_config.get("fan_out_max_workers", 8))

    @property
    def default_rerank_mode(self) -> str | None:
        """Default cross-encoder reranker slug for search, or None.

        When set, every search call that doesn't explicitly pass
        ``rerank=`` will use this reranker automatically. Useful for
        users who've installed one of the optional extras
        (``rerank-provence`` or ``rerank-bge``) and want it always-on.
        ``None`` (default) preserves the legacy no-rerank behavior
        byte-for-byte.
        """
        env_val = os.environ.get("MEMPALACE_DEFAULT_RERANK_MODE")
        if env_val:
            return env_val if env_val != "none" else None
        val = self._file_config.get("default_rerank_mode")
        return val if val and val != "none" else None

    @property
    def rerank_provence_prune(self) -> bool:
        """Whether the Provence reranker should emit pruned_text by default.

        Only meaningful when ``default_rerank_mode == "provence"``.
        Users who want reranking without per-token pruning can set this
        to False.
        """
        return bool(self._file_config.get("rerank_provence_prune", True))

    def save_embedding_config(self, *, default=None, enabled=None):
        """Persist ``default_embedding_model`` / ``enabled_embedding_models``.

        Used by ``mempalace models enable/disable/set-default``. Merges
        into whatever config.json exists, creating it if missing. Other
        keys are preserved.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        current = {}
        if self._config_file.exists():
            try:
                with open(self._config_file) as f:
                    current = json.load(f)
            except (json.JSONDecodeError, OSError):
                current = {}
        if default is not None:
            current["default_embedding_model"] = default
        if enabled is not None:
            current["enabled_embedding_models"] = list(enabled)
        with open(self._config_file, "w") as f:
            json.dump(current, f, indent=2)
        self._file_config = current
        return self._config_file

    def init(self):
        """Create config directory and write default config.json if it doesn't exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        # Restrict directory permissions to owner only (Unix)
        try:
            self._config_dir.chmod(0o700)
        except (OSError, NotImplementedError):
            pass  # Windows doesn't support Unix permissions
        if not self._config_file.exists():
            default_config = {
                "palace_path": DEFAULT_PALACE_PATH,
                "collection_name": DEFAULT_COLLECTION_NAME,
                "topic_wings": DEFAULT_TOPIC_WINGS,
                "hall_keywords": DEFAULT_HALL_KEYWORDS,
            }
            with open(self._config_file, "w") as f:
                json.dump(default_config, f, indent=2)
            # Restrict config file to owner read/write only
            try:
                self._config_file.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
        return self._config_file

    def save_people_map(self, people_map):
        """Write people_map.json to config directory.

        Args:
            people_map: Dict mapping name variants to canonical names.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._people_map_file, "w") as f:
            json.dump(people_map, f, indent=2)
        return self._people_map_file
