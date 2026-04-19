"""
test_aggregates.py — Unit tests for the hierarchical aggregate layer.

Covers:
  * ``classify_hall`` keyword routing and default fallback.
  * ``hydrate_drawer_metadata`` idempotency (preserves pre-set halls).
  * Dirty-set round-trip through the trie's meta DBI.
  * ``compute_aggregate_text`` edge cases: zero / one / N<=K / N>K.
  * Sentinel container exclusion (room=_registry, hall=hall_registry).
  * ``upsert_aggregate`` writes and deletes on empty text.
  * ``aggregate_contributions`` returns empty dict when disabled.

Everything runs offline with the real trie + stub Chroma collections.
"""

from unittest.mock import MagicMock

import pytest

from mempalace import aggregates
from mempalace.aggregates import (
    classify_hall,
    clear_dirty,
    compute_aggregate_text,
    hydrate_drawer_metadata,
    list_dirty,
    mark_container_dirty,
)

# ── classify_hall ────────────────────────────────────────────────────


class TestClassifyHall:
    def test_empty_text_returns_default(self):
        assert classify_hall("") == "hall_general"
        assert classify_hall("   ") == "hall_general"

    def test_unknown_content_falls_back_to_default(self):
        # "jabberwocky" matches no keyword in DEFAULT_HALL_KEYWORDS
        assert classify_hall("jabberwocky frumious bandersnatch") == "hall_general"

    def test_technical_keyword_hits_technical_hall(self):
        assert classify_hall("Debug the python api error in the server") == "hall_technical"

    def test_emotions_keyword_hits_emotions_hall(self):
        assert classify_hall("I feel so happy about this release") == "hall_emotions"

    def test_memory_keyword_hits_memory_hall(self):
        assert classify_hall("The palace stores every memory") == "hall_memory"

    def test_custom_default_is_honored(self):
        assert classify_hall("", default="hall_fallback") == "hall_fallback"

    def test_lowercasing(self):
        # Capitalized keyword should still match the lowercase one in
        # the keyword map.
        assert classify_hall("FUNCTION returns api response") == "hall_technical"


# ── hydrate_drawer_metadata ──────────────────────────────────────────


class TestHydrateDrawerMetadata:
    def test_sets_hall_when_missing(self):
        meta: dict = {"wing": "project", "room": "backend"}
        out = hydrate_drawer_metadata(meta, "The python script has a bug.")
        assert out["hall"] == "hall_technical"
        # hydration is in-place
        assert meta is out

    def test_preserves_existing_hall(self):
        meta = {"wing": "journal", "room": "diary", "hall": "hall_diary"}
        hydrate_drawer_metadata(meta, "I feel so happy today")
        assert meta["hall"] == "hall_diary"  # untouched despite emotion keyword

    def test_empty_content_falls_back_to_general(self):
        meta = {"wing": "any", "room": "any"}
        hydrate_drawer_metadata(meta, "")
        assert meta["hall"] == "hall_general"

    def test_does_not_touch_other_fields(self):
        meta = {"wing": "w", "room": "r", "extra": 42, "source_file": "a.py"}
        hydrate_drawer_metadata(meta, "function api")
        assert meta["wing"] == "w"
        assert meta["extra"] == 42
        assert meta["source_file"] == "a.py"


# ── Dirty tracking ───────────────────────────────────────────────────


class TestDirtyTracking:
    def test_mark_and_list_with_seeded_trie(self, palace_path, seeded_trie):
        mark_container_dirty(palace_path, wing="wing_alpha", room="room_x")
        mark_container_dirty(palace_path, hall="hall_technical")

        dirty = list_dirty(palace_path)
        assert "wing_alpha" in dirty["wing"]
        assert "hall_technical" in dirty["hall"]
        assert "room_x" in dirty["room"]

    def test_mark_dedupes(self, palace_path, seeded_trie):
        mark_container_dirty(palace_path, wing="wing_alpha")
        mark_container_dirty(palace_path, wing="wing_alpha")
        dirty = list_dirty(palace_path)
        assert dirty["wing"].count("wing_alpha") == 1

    def test_clear_removes_containers(self, palace_path, seeded_trie):
        mark_container_dirty(palace_path, wing="wing_alpha")
        mark_container_dirty(palace_path, wing="wing_beta")
        clear_dirty(palace_path, "wing", ["wing_alpha"])

        dirty = list_dirty(palace_path)
        assert "wing_alpha" not in dirty["wing"]
        assert "wing_beta" in dirty["wing"]

    def test_sentinel_containers_are_skipped(self, palace_path, seeded_trie):
        mark_container_dirty(palace_path, room="_registry")
        mark_container_dirty(palace_path, hall="hall_registry")
        mark_container_dirty(palace_path, room="diary")

        dirty = list_dirty(palace_path)
        assert "_registry" not in dirty["room"]
        assert "hall_registry" not in dirty["hall"]
        assert "diary" not in dirty["room"]

    def test_list_dirty_on_virgin_palace(self, palace_path, seeded_trie):
        # Fresh trie, no dirty writes — every level empty.
        dirty = list_dirty(palace_path)
        assert dirty == {"wing": [], "hall": [], "room": []}


# ── compute_aggregate_text ───────────────────────────────────────────


def _fake_col_get(pages):
    """Build a fake Chroma collection whose ``.get`` returns ``pages``.

    ``pages`` is a dict like ``{"ids": [...], "documents": [...],
    "embeddings": [...]}``. A MagicMock is returned so tests can
    assert on .get calls too.
    """
    col = MagicMock()
    col.get.return_value = pages
    return col


class TestComputeAggregateText:
    def test_zero_drawers_returns_empty(self):
        col = _fake_col_get({"ids": [], "documents": [], "embeddings": []})
        text, ids = compute_aggregate_text(col, level="room", container="room_x", top_k=5)
        assert text == ""
        assert ids == []

    def test_single_drawer_returns_that_drawer(self):
        col = _fake_col_get(
            {
                "ids": ["d1"],
                "documents": ["hello world"],
                "embeddings": [[1.0, 0.0, 0.0]],
            }
        )
        text, ids = compute_aggregate_text(col, level="room", container="room_x", top_k=5)
        assert text == "hello world"
        assert ids == ["d1"]

    def test_n_less_than_or_equal_to_top_k_keeps_all(self):
        col = _fake_col_get(
            {
                "ids": ["d1", "d2", "d3"],
                "documents": ["alpha beta", "gamma delta", "epsilon zeta"],
                "embeddings": [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
            }
        )
        text, ids = compute_aggregate_text(col, level="room", container="room_x", top_k=5)
        assert len(ids) == 3
        assert "alpha beta" in text
        assert "gamma delta" in text
        assert "epsilon zeta" in text

    def test_n_greater_than_top_k_picks_centroid_closest(self):
        # Four drawers; three cluster tight around [1,0], one outlier
        # at [0,1]. Centroid ≈ [0.775, 0.225], so the tight cluster
        # ranks highest and the outlier drops out when top_k=3.
        col = _fake_col_get(
            {
                "ids": ["a", "b", "c", "outlier"],
                "documents": ["Adoc", "Bdoc", "Cdoc", "Odoc"],
                "embeddings": [
                    [1.0, 0.0],
                    [0.95, 0.05],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
            }
        )
        text, ids = compute_aggregate_text(col, level="room", container="room_x", top_k=3)
        assert len(ids) == 3
        assert "outlier" not in ids
        # The tight cluster members must be present.
        for wanted in ("a", "b", "c"):
            assert wanted in ids

    def test_sentinel_container_returns_empty(self):
        # Room=_registry is sentinel; compute must short-circuit BEFORE
        # hitting the collection so the ``.get`` mock is never called.
        col = _fake_col_get({"ids": [], "documents": [], "embeddings": []})
        text, ids = compute_aggregate_text(col, level="room", container="_registry", top_k=5)
        assert text == ""
        assert ids == []
        col.get.assert_not_called()

    def test_skips_empty_documents(self):
        col = _fake_col_get(
            {
                "ids": ["d1", "d2"],
                "documents": ["", "real content"],
                "embeddings": [[1.0, 0.0], [0.5, 0.5]],
            }
        )
        text, ids = compute_aggregate_text(col, level="room", container="room_x", top_k=5)
        assert ids == ["d2"]
        assert text == "real content"

    def test_missing_embeddings_falls_back_to_first_k(self):
        # When embeddings are missing for all entries, compute should
        # still produce a deterministic top_k by taking the first K.
        col = _fake_col_get(
            {
                "ids": ["a", "b", "c", "d"],
                "documents": ["A", "B", "C", "D"],
                "embeddings": [None, None, None, None],
            }
        )
        text, ids = compute_aggregate_text(col, level="room", container="room_x", top_k=2)
        assert ids == ["a", "b"]
        assert "A" in text
        assert "B" in text

    def test_invalid_level_raises(self):
        col = _fake_col_get({"ids": [], "documents": [], "embeddings": []})
        with pytest.raises(ValueError, match="level must be one of"):
            compute_aggregate_text(col, level="drawer", container="x", top_k=5)


# ── aggregate_contributions ──────────────────────────────────────────


class TestAggregateContributions:
    def test_returns_empty_when_disabled(self, palace_path, monkeypatch):
        monkeypatch.setenv("MEMPALACE_AGGREGATE_ENABLED", "false")
        out = aggregates.aggregate_contributions(
            "some query",
            palace_path,
            slug="default",
        )
        assert out == {}

    def test_returns_empty_when_query_blank(self, palace_path):
        out = aggregates.aggregate_contributions("", palace_path, slug="default")
        assert out == {}

    def test_returns_empty_for_fan_out_slug(self, palace_path):
        out = aggregates.aggregate_contributions("hello", palace_path, slug="all")
        assert out == {}

    def test_returns_empty_when_no_aggregate_collection_yet(
        self, palace_path, seeded_collection, monkeypatch
    ):
        monkeypatch.setenv("MEMPALACE_AGGREGATE_ENABLED", "true")
        # No aggregates built yet — helper should degrade silently.
        out = aggregates.aggregate_contributions(
            "authentication",
            palace_path,
            slug="default",
        )
        assert out == {}


# ── Sentinel constants sanity check ──────────────────────────────────


class TestSentinels:
    def test_registry_sentinels_present(self):
        assert "_registry" in aggregates.SENTINEL_ROOMS
        assert "hall_registry" in aggregates.SENTINEL_HALLS
