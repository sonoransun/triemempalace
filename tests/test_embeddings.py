"""
test_embeddings.py — Registry + lazy-load tests for the embedding module.

The core goals of the embeddings module are:

1. Zero-cost import (no heavy backend imported at module scope).
2. Lazy weight load (the model isn't touched until first __call__).
3. Stable slug → collection name mapping, including the legacy
   "default" → "mempalace_drawers" alias.
4. Correct ``is_installed`` probing via ``importlib.util.find_spec`` —
   no real import.
5. Idempotent ``load_embedding_function`` so every caller in the
   process shares the same adapter instance.

None of these tests require real model weights. A fake backend is
registered at setup time with a deterministic hash-based embedding
function so tests can exercise the coexistence / resolution code paths
without hitting the network or the filesystem.
"""

from __future__ import annotations

import hashlib
import sys

import pytest

from mempalace import embeddings
from mempalace.embeddings import (
    EmbeddingSpec,
    collection_name_for,
    get_spec,
    is_installed,
    list_specs,
    load_embedding_function,
    normalize_slug_for_collection,
    register,
)

# ── Fake backend plumbing ────────────────────────────────────────────


class _FakeFn:
    """Deterministic fake embedding function for tests.

    Hash-based, so identical text always produces the same vector, and
    two different texts produce uncorrelated vectors. 32-byte SHA256
    digest → 32 floats in [0, 1). The dimensionality doesn't need to
    match any real model — Chroma doesn't care as long as it's
    consistent within a collection.
    """

    DIM = 32

    def __init__(self, spec: EmbeddingSpec, tag: str = ""):
        self.spec = spec
        self.tag = tag
        self.calls = 0

    def __call__(self, input: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in input:
            h = hashlib.sha256((self.tag + text).encode("utf-8")).digest()
            vec = [b / 255.0 for b in h[: self.DIM]]
            out.append(vec)
        return out

    def name(self) -> str:
        return f"fake:{self.spec.slug}"


def _install_fake_backend(monkeypatch, slug: str, *, tag: str = ""):
    """Register a fake spec and patch load_embedding_function to return
    a ``_FakeFn`` instance for that slug only.

    Uses monkeypatch so the changes are reverted automatically at
    teardown and don't leak into other tests.
    """
    spec = EmbeddingSpec(
        slug=slug,
        display_name=f"Fake {slug}",
        description="Test-only deterministic hash embedder",
        backend="fake-test",
        model_id=f"fake/{slug}",
        dimension=_FakeFn.DIM,
        context_tokens=256,
        extras_required=(),
    )
    register(spec)

    # Patch load_embedding_function so the registry dispatch doesn't try
    # to import a real backend for this slug.
    original_load = embeddings.load_embedding_function

    def _patched_load(s: str):
        if s == slug:
            cached = embeddings._loaded_functions.get(s)
            if cached is not None:
                return cached
            fn = _FakeFn(spec, tag=tag)
            embeddings._loaded_functions[s] = fn
            return fn
        return original_load(s)

    monkeypatch.setattr(embeddings, "load_embedding_function", _patched_load)
    return spec


@pytest.fixture(autouse=True)
def _reset_embedding_cache():
    """Clear the process-wide loaded-function cache between tests."""
    embeddings.clear_cache()
    yield
    embeddings.clear_cache()


# ── Zero-cost import tests ───────────────────────────────────────────


class TestLazyImport:
    def test_registry_imports_without_heavy_backends(self):
        """Constructing specs and listing the registry must not import
        fastembed / sentence_transformers / ollama / torch."""
        heavy = {"fastembed", "sentence_transformers", "ollama", "torch", "transformers"}
        # Snapshot current modules, then re-touch the public API.
        before = set(sys.modules)
        specs = list_specs()
        assert len(specs) >= 7  # seven v1 specs
        after = set(sys.modules)
        leaked = heavy & (after - before)
        assert not leaked, f"Heavy backends leaked on list_specs(): {leaked}"

    def test_default_load_is_none_passthrough(self):
        """The 'default' slug must return None so callers omit
        embedding_function= and Chroma uses its built-in."""
        assert load_embedding_function("default") is None

    def test_is_installed_uses_find_spec(self):
        """is_installed probes importlib.util.find_spec without importing."""
        # The ``default`` spec has no extras → always installed.
        assert is_installed(get_spec("default")) is True

        # A spec with extras that almost certainly aren't installed.
        spec = EmbeddingSpec(
            slug="test-missing",
            display_name="Test missing",
            description="",
            backend="fake-test",
            model_id="x",
            dimension=32,
            context_tokens=256,
            extras_required=("definitely_not_an_installed_package_xyz",),
        )
        assert is_installed(spec) is False

    def test_missing_extras_raises_friendly_error(self):
        """load_embedding_function on a spec whose extras are missing
        must raise a RuntimeError with a pip-install hint."""
        # jina-code-v2 requires fastembed; the dev venv may or may not
        # have it. Register a clearly-missing fake spec instead.
        spec = EmbeddingSpec(
            slug="test-missing-real",
            display_name="Test missing real",
            description="",
            backend="fastembed",
            model_id="x",
            dimension=32,
            context_tokens=256,
            extras_required=("definitely_not_installed_pkg_abc",),
        )
        register(spec)
        with pytest.raises(RuntimeError, match="pip install"):
            load_embedding_function("test-missing-real")


# ── Collection naming ────────────────────────────────────────────────


class TestCollectionNaming:
    def test_default_slug_maps_to_legacy_name(self):
        """The ``default`` slug must return the legacy collection name
        so existing palaces continue to work byte-for-byte."""
        assert collection_name_for("default") == "mempalace_drawers"

    def test_non_default_slugs_get_prefix(self):
        assert collection_name_for("jina-code-v2") == "mempalace_drawers__jina_code_v2"
        assert collection_name_for("nomic-text-v1.5") == "mempalace_drawers__nomic_text_v1_5"
        assert collection_name_for("mxbai-large") == "mempalace_drawers__mxbai_large"

    def test_slug_normalizer_strips_punctuation(self):
        assert normalize_slug_for_collection("nomic-text-v1.5") == "nomic_text_v1_5"
        assert normalize_slug_for_collection("BGE-Small-EN") == "bge_small_en"
        # Leading/trailing underscores get stripped.
        assert normalize_slug_for_collection("--weird--") == "weird"


# ── Registry CRUD ────────────────────────────────────────────────────


class TestRegistry:
    def test_list_includes_all_v1_specs(self):
        slugs = {s.slug for s in list_specs()}
        expected = {
            "default",
            "bge-small-en",
            "nomic-text-v1.5",
            "jina-code-v2",
            "mxbai-large",
            "ollama-nomic",
            "ollama-mxbai",
        }
        assert expected <= slugs

    def test_get_spec_unknown_slug_raises(self):
        with pytest.raises(KeyError, match="Unknown embedding model"):
            get_spec("totally-not-a-real-slug-zzz")

    def test_register_adds_new_spec(self):
        spec = EmbeddingSpec(
            slug="test-extra-slug",
            display_name="Extra",
            description="",
            backend="fake-test",
            model_id="x",
            dimension=32,
            context_tokens=100,
            extras_required=(),
        )
        register(spec)
        assert get_spec("test-extra-slug") is spec


# ── Fake-backend coexistence tests ───────────────────────────────────


class TestFakeBackendCoexistence:
    """End-to-end: two fake models coexist in one palace without
    stepping on each other."""

    def test_fake_fn_is_idempotent(self, monkeypatch, tmp_path):
        # Use attribute access (embeddings.load_embedding_function) so the
        # monkeypatch is actually observed — ``from ... import`` binds the
        # original function into the test module's namespace.
        spec = _install_fake_backend(monkeypatch, "test-fake-a", tag="A")
        fn1 = embeddings.load_embedding_function("test-fake-a")
        fn2 = embeddings.load_embedding_function("test-fake-a")
        assert fn1 is fn2, "load_embedding_function must memoize per slug"
        assert isinstance(fn1, _FakeFn)
        assert fn1.spec.slug == spec.slug

    def test_two_fake_models_produce_disjoint_vectors(self, monkeypatch):
        _install_fake_backend(monkeypatch, "test-fake-x", tag="X")
        _install_fake_backend(monkeypatch, "test-fake-y", tag="Y")
        fn_x = embeddings.load_embedding_function("test-fake-x")
        fn_y = embeddings.load_embedding_function("test-fake-y")
        vec_x = fn_x(["hello"])[0]
        vec_y = fn_y(["hello"])[0]
        assert vec_x != vec_y, "tagged fake backends must produce different vectors"

    def test_fake_backend_via_open_collection(self, monkeypatch, tmp_path):
        """The palace_io.open_collection seam must pick up fake specs
        and return a Chroma collection that uses them."""
        from mempalace import palace_io

        _install_fake_backend(monkeypatch, "test-fake-coexist-a", tag="A")
        _install_fake_backend(monkeypatch, "test-fake-coexist-b", tag="B")

        palace = str(tmp_path / "palace")

        col_a = palace_io.open_collection(palace, model="test-fake-coexist-a", create=True)
        col_b = palace_io.open_collection(palace, model="test-fake-coexist-b", create=True)
        assert col_a is not col_b, "different models must land in different collections"
        assert col_a.name != col_b.name

        # Write a drawer into each collection and confirm they don't
        # cross-contaminate.
        col_a.add(
            ids=["drawer_a"],
            documents=["alpha beta"],
            metadatas=[{"wing": "w", "room": "r"}],
        )
        col_b.add(
            ids=["drawer_b"],
            documents=["gamma delta"],
            metadatas=[{"wing": "w", "room": "r"}],
        )

        got_a = col_a.get(ids=["drawer_a"])
        got_b = col_b.get(ids=["drawer_b"])
        assert got_a["ids"] == ["drawer_a"]
        assert got_b["ids"] == ["drawer_b"]

        # Drawer from collection A must not appear in collection B.
        missing_b = col_b.get(ids=["drawer_a"])
        assert missing_b["ids"] == []

        palace_io.close_all()


# ── Smoke: stats are reachable via MempalaceConfig ────────────────────


class TestConfigIntegration:
    def test_defaults_on_fresh_install(self, tmp_path):
        """MempalaceConfig without a config.json returns the defaults."""
        from mempalace.config import MempalaceConfig

        cfg = MempalaceConfig(config_dir=str(tmp_path / "empty_cfg"))
        assert cfg.default_embedding_model == "default"
        assert cfg.enabled_embedding_models == ["default"]

    def test_env_var_override(self, monkeypatch, tmp_path):
        from mempalace.config import MempalaceConfig

        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "jina-code-v2")
        cfg = MempalaceConfig(config_dir=str(tmp_path / "empty_cfg2"))
        assert cfg.default_embedding_model == "jina-code-v2"

    def test_save_embedding_config_round_trips(self, tmp_path):
        from mempalace.config import MempalaceConfig

        cfg_dir = tmp_path / "cfg"
        cfg = MempalaceConfig(config_dir=str(cfg_dir))
        cfg.save_embedding_config(default="nomic-text-v1.5", enabled=["default", "nomic-text-v1.5"])

        # Reopen — values persist.
        cfg2 = MempalaceConfig(config_dir=str(cfg_dir))
        assert cfg2.default_embedding_model == "nomic-text-v1.5"
        assert cfg2.enabled_embedding_models == ["default", "nomic-text-v1.5"]


class TestMatryoshkaTruncation:
    """Matryoshka (MRL) dimension truncation support on EmbeddingSpec.

    These tests exercise the ``_apply_matryoshka`` helper directly and
    via the fastembed/sentence-transformers adapters, without ever
    touching real model weights. The adapter paths use monkeypatched
    ``_ensure`` methods so no backend is imported.
    """

    def test_truncate_dim_none_is_passthrough(self):
        """No truncation when truncate_dim is unset."""
        from mempalace.embeddings import _apply_matryoshka

        spec = EmbeddingSpec(
            slug="test-no-trunc",
            display_name="No truncation",
            description="",
            backend="fake-test",
            model_id="fake",
            dimension=16,
            context_tokens=128,
            extras_required=(),
            supports_matryoshka=True,
            truncate_dim=None,
        )
        vectors = [[0.1] * 16, [0.2] * 16]
        result = _apply_matryoshka(vectors, spec)
        assert result == vectors
        assert len(result[0]) == 16

    def test_truncate_dim_slices_vectors(self):
        """Setting truncate_dim=8 on a 16-dim spec yields 8-dim vectors."""
        from mempalace.embeddings import _apply_matryoshka

        spec = EmbeddingSpec(
            slug="test-trunc",
            display_name="Truncated",
            description="",
            backend="fake-test",
            model_id="fake",
            dimension=16,
            context_tokens=128,
            extras_required=(),
            supports_matryoshka=True,
            truncate_dim=8,
        )
        vectors = [list(range(16)), list(range(100, 116))]
        result = _apply_matryoshka(vectors, spec)
        assert len(result) == 2
        assert len(result[0]) == 8
        assert len(result[1]) == 8
        # First 8 dims preserved exactly.
        assert result[0] == list(range(8))
        assert result[1] == list(range(100, 108))

    def test_truncate_without_supports_flag_raises(self):
        """Truncating a non-MRL spec is a corruption bug — must raise."""
        from mempalace.embeddings import _apply_matryoshka

        spec = EmbeddingSpec(
            slug="test-non-mrl",
            display_name="Non-MRL",
            description="",
            backend="fake-test",
            model_id="fake",
            dimension=16,
            context_tokens=128,
            extras_required=(),
            supports_matryoshka=False,  # explicitly False
            truncate_dim=8,
        )
        with pytest.raises(ValueError, match="supports_matryoshka=False"):
            _apply_matryoshka([[0.1] * 16], spec)

    def test_fastembed_adapter_applies_truncation(self, monkeypatch):
        """The _FastEmbedFn adapter slices before returning."""
        from mempalace.embeddings import _FastEmbedFn

        spec = EmbeddingSpec(
            slug="fake-fastembed-trunc",
            display_name="Fake fastembed trunc",
            description="",
            backend="fastembed",
            model_id="fake/fastembed",
            dimension=16,
            context_tokens=128,
            extras_required=("fastembed",),
            supports_matryoshka=True,
            truncate_dim=4,
        )
        adapter = _FastEmbedFn(spec)

        class _FakeFastEmbedModel:
            def embed(self, texts):

                # fastembed returns numpy arrays — simulate .tolist()
                class _FakeVec:
                    def __init__(self, data):
                        self._data = data

                    def tolist(self):
                        return list(self._data)

                for i, _ in enumerate(texts):
                    yield _FakeVec([float(i)] * 16)

        # Bypass the real fastembed import
        adapter._model = _FakeFastEmbedModel()

        result = adapter(["a", "b"])
        assert len(result) == 2
        assert len(result[0]) == 4  # truncated from 16
        assert result[0] == [0.0, 0.0, 0.0, 0.0]
        assert result[1] == [1.0, 1.0, 1.0, 1.0]

    def test_known_mrl_models_flagged(self):
        """The three registered MRL-trained models carry the flag."""
        # nomic-text-v1.5, mxbai-large, bge-m3 are all trained with MRL.
        # jina-code-v2 is NOT MRL-trained and must stay False.
        nomic = get_spec("nomic-text-v1.5")
        mxbai = get_spec("mxbai-large")
        bge_m3 = get_spec("bge-m3")
        jina = get_spec("jina-code-v2")

        assert nomic.supports_matryoshka is True
        assert mxbai.supports_matryoshka is True
        assert bge_m3.supports_matryoshka is True
        assert jina.supports_matryoshka is False

    def test_bge_m3_spec_shape(self):
        """BGE-M3 spec matches its paper: 1024-dim, 8192-token context, fastembed."""
        spec = get_spec("bge-m3")
        assert spec.backend == "fastembed"
        assert spec.model_id == "BAAI/bge-m3"
        assert spec.dimension == 1024
        assert spec.context_tokens == 8192
        assert spec.supports_matryoshka is True


class TestHnswEfSearch:
    """Config-level HNSW ef_search tuning."""

    def test_default_is_forty(self, tmp_path):
        """Fresh install returns the DEFAULT_HNSW_EF_SEARCH constant (40)."""
        from mempalace.config import DEFAULT_HNSW_EF_SEARCH, MempalaceConfig

        cfg = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert cfg.hnsw_ef_search == DEFAULT_HNSW_EF_SEARCH == 40

    def test_env_var_override(self, monkeypatch, tmp_path):
        from mempalace.config import MempalaceConfig

        monkeypatch.setenv("MEMPALACE_HNSW_EF_SEARCH", "128")
        cfg = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert cfg.hnsw_ef_search == 128

    def test_config_file_override(self, tmp_path):
        import json

        from mempalace.config import MempalaceConfig

        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(json.dumps({"hnsw_ef_search": 80}))

        cfg = MempalaceConfig(config_dir=str(cfg_dir))
        assert cfg.hnsw_ef_search == 80

    def test_env_beats_file(self, monkeypatch, tmp_path):
        import json

        from mempalace.config import MempalaceConfig

        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(json.dumps({"hnsw_ef_search": 80}))
        monkeypatch.setenv("MEMPALACE_HNSW_EF_SEARCH", "200")

        cfg = MempalaceConfig(config_dir=str(cfg_dir))
        assert cfg.hnsw_ef_search == 200
