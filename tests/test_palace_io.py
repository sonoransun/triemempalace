"""
test_palace_io.py — Tests for the single ChromaDB seam.

palace_io is the only module in the package allowed to construct a
``chromadb.PersistentClient``. Every other writer and reader funnels
through ``open_collection``, so the seam absolutely needs coverage of:

* Default-model resolution from ``MempalaceConfig``.
* Per-(palace, collection_name) caching.
* The ``create=True`` path for fresh palaces.
* The ``collection_name_override`` escape hatch used by
  ``mempalace compress`` for the ``mempalace_compressed`` sidecar.
* ``delete_collection`` + ``drop_collection_cache`` invalidation.
* ``close_all`` resetting both the client and collection caches.
* Rejection of ``model="all"`` (a query-time fan-out flag, never a
  real collection name).

These tests use the same ``_install_fake_backend`` helper from
``tests/test_embeddings.py`` so CI never downloads real model weights.
"""

import pytest

from mempalace import embeddings, palace_io
from tests.test_embeddings import _install_fake_backend  # noqa: F401  (re-used helper)


@pytest.fixture(autouse=True)
def _reset_palace_io():
    """Drop every cached client + collection between tests."""
    palace_io.close_all()
    embeddings.clear_cache()
    yield
    palace_io.close_all()
    embeddings.clear_cache()


# ── default model resolution ─────────────────────────────────────────


def test_resolve_model_falls_back_to_config_default(monkeypatch):
    """``_resolve_model(None)`` returns the config's default slug."""

    class _Stub:
        default_embedding_model = "default"

    monkeypatch.setattr("mempalace.config.MempalaceConfig", lambda: _Stub())
    assert palace_io._resolve_model(None) == "default"
    assert palace_io._resolve_model("explicit-slug") == "explicit-slug"


def test_canonical_normalizes_path(tmp_path):
    """``_canonical`` expanduser + absolute, but does NOT follow symlinks."""
    canon = palace_io._canonical(tmp_path)
    assert canon == str(tmp_path.absolute())


# ── caching behavior ─────────────────────────────────────────────────


def test_open_collection_returns_same_handle_on_repeat_calls(tmp_path):
    """A second call with the same args hits the cache."""
    palace = str(tmp_path / "palace")
    first = palace_io.open_collection(palace, model="default", create=True)
    second = palace_io.open_collection(palace, model="default")
    assert first is second


def test_open_collection_create_true_makes_fresh_collection(tmp_path):
    """``create=True`` succeeds on a brand-new palace dir."""
    palace = str(tmp_path / "palace")
    col = palace_io.open_collection(palace, model="default", create=True)
    assert col is not None
    assert col.count() == 0


def test_close_all_invalidates_handles(tmp_path):
    """After ``close_all`` a fresh ``open_collection`` returns a new object."""
    palace = str(tmp_path / "palace")
    first = palace_io.open_collection(palace, model="default", create=True)
    palace_io.close_all()
    second = palace_io.open_collection(palace, model="default", create=True)
    # Different cached object after close_all (the underlying Chroma data is the same).
    assert first is not second


def test_drop_collection_cache_targets_one_palace(tmp_path):
    """``drop_collection_cache`` clears handles for a specific palace path."""
    palace = str(tmp_path / "palace")
    col = palace_io.open_collection(palace, model="default", create=True)
    assert col is not None

    palace_io.drop_collection_cache(palace)

    refreshed = palace_io.open_collection(palace, model="default")
    assert refreshed is not col


# ── error paths ──────────────────────────────────────────────────────


def test_open_collection_rejects_model_all(tmp_path):
    """``model='all'`` is a fan-out flag at query time, not a real collection."""
    palace = str(tmp_path / "palace")
    with pytest.raises(ValueError, match="fan-out"):
        palace_io.open_collection(palace, model="all", create=True)


def test_open_collection_unknown_slug_raises(tmp_path):
    """An unknown slug propagates the registry's KeyError as a clean failure."""
    palace = str(tmp_path / "palace")
    # ``embeddings.collection_name_for`` raises KeyError for unknown slugs.
    with pytest.raises(KeyError):
        palace_io.open_collection(palace, model="not-a-real-slug", create=True)


# ── collection_name_override (compress sidecar) ──────────────────────


def test_open_collection_with_override_uses_literal_name(tmp_path):
    """``collection_name_override`` bypasses the slug→name lookup."""
    palace = str(tmp_path / "palace")
    col = palace_io.open_collection(
        palace,
        create=True,
        collection_name_override="mempalace_compressed",
    )
    assert col.name == "mempalace_compressed"


def test_open_collection_override_caches_independently(tmp_path):
    """The override key is namespaced separately from the default-model handle."""
    palace = str(tmp_path / "palace")
    default_col = palace_io.open_collection(palace, model="default", create=True)
    override_col = palace_io.open_collection(
        palace,
        create=True,
        collection_name_override="mempalace_compressed",
    )
    assert default_col is not override_col
    assert override_col.name == "mempalace_compressed"
    # The override hits the cache on a second call.
    again = palace_io.open_collection(palace, collection_name_override="mempalace_compressed")
    assert again is override_col


def test_open_collection_override_ignores_model(tmp_path):
    """When ``collection_name_override`` is set, ``model`` is ignored entirely."""
    palace = str(tmp_path / "palace")
    col = palace_io.open_collection(
        palace,
        model="not-a-real-slug",  # would normally raise
        create=True,
        collection_name_override="mempalace_compressed",
    )
    assert col.name == "mempalace_compressed"


# ── delete_collection helper ─────────────────────────────────────────


def test_delete_collection_drops_default_model(tmp_path):
    """``delete_collection`` removes the named collection and clears its cache."""
    palace = str(tmp_path / "palace")
    col = palace_io.open_collection(palace, model="default", create=True)
    col.add(ids=["d1"], documents=["alpha"], metadatas=[{"wing": "w"}])
    assert col.count() == 1

    palace_io.delete_collection(palace, model="default")

    # A re-create returns an empty collection (the data is gone).
    fresh = palace_io.open_collection(palace, model="default", create=True)
    assert fresh.count() == 0


def test_delete_collection_with_override(tmp_path):
    """``delete_collection`` honors ``collection_name_override`` symmetrically with open."""
    palace = str(tmp_path / "palace")
    palace_io.open_collection(palace, create=True, collection_name_override="mempalace_compressed")
    palace_io.delete_collection(palace, collection_name_override="mempalace_compressed")
    # Re-creating succeeds — the previous one is gone.
    fresh = palace_io.open_collection(
        palace, create=True, collection_name_override="mempalace_compressed"
    )
    assert fresh.count() == 0


def test_delete_collection_rejects_model_all(tmp_path):
    """``delete_collection`` matches ``open_collection``'s rejection of ``model='all'``."""
    palace = str(tmp_path / "palace")
    with pytest.raises(ValueError, match="fan-out"):
        palace_io.delete_collection(palace, model="all")


# ── multi-model coexistence ──────────────────────────────────────────


def test_open_collection_multiple_models_in_one_palace(tmp_path, monkeypatch):
    """Two different model slugs map to two distinct collections in the same palace."""
    palace = str(tmp_path / "palace")
    _install_fake_backend(monkeypatch, "fake-mini")

    default_col = palace_io.open_collection(palace, model="default", create=True)
    fake_col = palace_io.open_collection(palace, model="fake-mini", create=True)

    assert default_col is not fake_col
    assert default_col.name == "mempalace_drawers"
    # Fake slug uses the namespaced collection name from collection_name_for.
    assert fake_col.name != "mempalace_drawers"

    # Adding to one doesn't affect the other.
    default_col.add(ids=["d1"], documents=["one"], metadatas=[{"w": "x"}])
    assert default_col.count() == 1
    assert fake_col.count() == 0
