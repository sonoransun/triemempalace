"""
test_trie_index.py — Unit tests for the keyword + temporal trie index.

Mirrors the style of ``test_knowledge_graph.py`` — uses the ``trie`` and
``seeded_trie`` fixtures from ``conftest.py`` for isolation.
"""

import os
from datetime import UTC

from mempalace.trie_index import TrieIndex, tokenize

# ── Tokenizer ──────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic_words(self):
        tokens = tokenize("the quick brown fox jumped")
        # "the" is a stopword; others survive
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens

    def test_lowercases(self):
        tokens = tokenize("JWT Authentication")
        assert "jwt" in tokens
        assert "authentication" in tokens

    def test_preserves_identifiers(self):
        tokens = tokenize("we use oauth2 and gpt-4 alongside snake_case_name")
        assert "oauth2" in tokens
        assert "gpt-4" in tokens
        assert "snake_case_name" in tokens

    def test_drops_short_tokens(self):
        # Single-character runs get dropped.
        tokens = tokenize("a b c de fg")
        assert "de" in tokens
        assert "fg" in tokens

    def test_empty_string(self):
        assert tokenize("") == []


# ── Write / read round-trip ────────────────────────────────────────────


class TestAddAndLookup:
    def test_add_drawer_inserts_tokens(self, trie):
        n = trie.add_drawer(
            "d1",
            "Database migrations use Alembic for PostgreSQL.",
            {"wing": "project", "room": "backend", "filed_at": "2026-01-02T00:00:00"},
        )
        assert n > 0

        hits = trie.lookup("alembic")
        assert "d1" in hits
        hits = trie.lookup("postgresql")
        assert "d1" in hits

    def test_case_insensitive_lookup(self, trie):
        trie.add_drawer(
            "d1",
            "JWT Authentication",
            {"wing": "w", "room": "r", "filed_at": "2026-01-01T00:00:00"},
        )
        assert "d1" in trie.lookup("jwt")
        assert "d1" in trie.lookup("JWT")
        assert "d1" in trie.lookup("Jwt")

    def test_prefix_lookup(self, trie):
        trie.add_drawer("d1", "authentication module", {"filed_at": "2026-01-01T00:00:00"})
        trie.add_drawer("d2", "author of the doc", {"filed_at": "2026-01-01T00:00:00"})
        hits = trie.lookup("auth", prefix=True)
        assert "d1" in hits
        assert "d2" in hits

        exact = trie.lookup("auth")
        assert "d1" not in exact
        assert "d2" not in exact

    def test_unknown_token_returns_empty(self, trie):
        trie.add_drawer("d1", "alpha beta gamma", {"filed_at": "2026-01-01T00:00:00"})
        assert trie.lookup("xyzzy_never_seen") == set()
        # Sanity: the trie DID index something.
        assert "d1" in trie.lookup("alpha")

    def test_delete_drawer_removes_postings(self, trie):
        trie.add_drawer("d1", "alembic migrations", {"filed_at": "2026-01-01T00:00:00"})
        assert "d1" in trie.lookup("alembic")

        trie.delete_drawer("d1")
        assert trie.lookup("alembic") == set()

    def test_add_batch_is_equivalent(self, trie):
        trie.add_batch(
            [
                ("d1", "alpha beta", {"filed_at": "2026-01-01T00:00:00"}),
                ("d2", "beta gamma", {"filed_at": "2026-01-02T00:00:00"}),
            ]
        )
        assert trie.lookup("alpha") == {"d1"}
        assert trie.lookup("beta") == {"d1", "d2"}
        assert trie.lookup("gamma") == {"d2"}


# ── Persistence ────────────────────────────────────────────────────────


class TestPersistence:
    def test_state_survives_reopen(self, tmp_dir):
        path = os.path.join(tmp_dir, "trie.sqlite3")
        t1 = TrieIndex(db_path=path)
        t1.add_drawer(
            "d1",
            "JWT tokens expire after 24 hours",
            {"wing": "proj", "room": "auth", "filed_at": "2026-01-01T00:00:00"},
        )
        del t1

        t2 = TrieIndex(db_path=path)
        assert "d1" in t2.lookup("jwt")
        stats = t2.stats()
        assert stats["postings"] > 0
        assert stats["unique_drawers"] == 1

    def test_stats_counts_unique(self, trie):
        trie.add_batch(
            [
                ("d1", "alpha alpha alpha beta", {"filed_at": "2026-01-01"}),
                ("d2", "alpha gamma", {"filed_at": "2026-01-02"}),
            ]
        )
        stats = trie.stats()
        assert stats["unique_drawers"] == 2
        # "alpha" appears in d1 and d2 = 2 postings, beta = 1, gamma = 1
        assert stats["unique_tokens"] == 3
        assert stats["postings"] == 4


# ── Scope filters ──────────────────────────────────────────────────────


class TestScopeFilters:
    def test_wing_filter(self, trie):
        trie.add_drawer(
            "d1",
            "authentication token",
            {"wing": "project", "room": "backend", "filed_at": "2026-01-01"},
        )
        trie.add_drawer(
            "d2",
            "authentication flow",
            {"wing": "notes", "room": "backend", "filed_at": "2026-01-01"},
        )
        hits = trie.lookup("authentication", wing="project")
        assert hits == {"d1"}

    def test_room_filter(self, trie):
        trie.add_drawer(
            "d1",
            "alpha beta",
            {"wing": "w", "room": "backend", "filed_at": "2026-01-01"},
        )
        trie.add_drawer(
            "d2",
            "alpha beta",
            {"wing": "w", "room": "frontend", "filed_at": "2026-01-01"},
        )
        assert trie.lookup("alpha", room="backend") == {"d1"}
        assert trie.lookup("alpha", room="frontend") == {"d2"}


# ── Temporal filters ───────────────────────────────────────────────────


class TestTemporal:
    def _seed(self, trie):
        trie.add_drawer(
            "d_old",
            "migration plan discussed",
            {"wing": "proj", "room": "planning", "filed_at": "2025-06-15T00:00:00"},
        )
        trie.add_drawer(
            "d_mid",
            "migration underway",
            {"wing": "proj", "room": "planning", "filed_at": "2026-01-10T00:00:00"},
        )
        trie.add_drawer(
            "d_new",
            "migration complete",
            {"wing": "proj", "room": "planning", "filed_at": "2026-03-20T00:00:00"},
        )

    def test_since_filters_out_older(self, trie):
        self._seed(trie)
        hits = trie.lookup("migration", since="2026-01-01")
        assert hits == {"d_mid", "d_new"}

    def test_until_filters_out_newer(self, trie):
        self._seed(trie)
        hits = trie.lookup("migration", until="2026-02-01")
        assert hits == {"d_old", "d_mid"}

    def test_since_and_until_window(self, trie):
        self._seed(trie)
        hits = trie.lookup("migration", since="2026-01-01", until="2026-02-01")
        assert hits == {"d_mid"}

    def test_as_of_with_explicit_bounds(self, trie):
        # Simulate a fact with a finite validity window.
        trie.add_drawer(
            "d_old_job",
            "Alice works at Acme",
            {"wing": "w", "room": "people", "filed_at": "2020-01-01"},
            valid_from="2020-01-01",
            valid_to="2024-12-31",
        )
        trie.add_drawer(
            "d_new_job",
            "Alice works at NewCo",
            {"wing": "w", "room": "people", "filed_at": "2025-01-01"},
            valid_from="2025-01-01",
        )

        # Ask "what was true on 2023-06-01?"
        hits = trie.lookup("alice", as_of="2023-06-01")
        assert "d_old_job" in hits
        assert "d_new_job" not in hits

        hits = trie.lookup("alice", as_of="2026-01-01")
        assert "d_old_job" not in hits
        assert "d_new_job" in hits

    def test_pure_temporal_query_via_keyword_search(self, trie):
        self._seed(trie)
        # keywords=[] → pure scope/temporal query
        hits = trie.keyword_search([], since="2026-01-01", wing="proj")
        assert hits == {"d_mid", "d_new"}


# ── keyword_search combine modes ───────────────────────────────────────


class TestKeywordSearch:
    def _seed(self, trie):
        trie.add_drawer(
            "d1",
            "jwt authentication flow",
            {"wing": "w", "room": "auth", "filed_at": "2026-01-01"},
        )
        trie.add_drawer(
            "d2",
            "oauth refresh token",
            {"wing": "w", "room": "auth", "filed_at": "2026-01-02"},
        )
        trie.add_drawer(
            "d3",
            "jwt refresh flow",
            {"wing": "w", "room": "auth", "filed_at": "2026-01-03"},
        )

    def test_mode_all_intersects(self, trie):
        self._seed(trie)
        hits = trie.keyword_search(["jwt", "refresh"], mode="all")
        assert hits == {"d3"}

    def test_mode_any_unions(self, trie):
        self._seed(trie)
        hits = trie.keyword_search(["oauth", "jwt"], mode="any")
        assert hits == {"d1", "d2", "d3"}

    def test_mode_prefix_intersects(self, trie):
        self._seed(trie)
        hits = trie.keyword_search(["auth", "fl"], mode="prefix")
        # "auth" matches d1 (authentication) and d2 (oauth)? oauth contains
        # no prefix "auth" starting at char 0 — tries match from the start
        # of the token, not the middle, so oauth is out.
        # "fl" matches "flow" → d1, d3.
        # Intersection: d1 (has "authentication" and "flow").
        assert hits == {"d1"}

    def test_mode_all_short_circuits_on_empty(self, trie):
        self._seed(trie)
        # First keyword has no hits → result should be empty immediately.
        hits = trie.keyword_search(["xyzzy", "jwt"], mode="all")
        assert hits == set()


# ── rebuild_from_collection ────────────────────────────────────────────


class TestRebuildFromCollection:
    def test_rebuilds_from_seeded_palace(self, trie, seeded_collection):
        total = trie.rebuild_from_collection(seeded_collection)
        assert total > 0

        # Every seeded drawer must be discoverable.
        assert "drawer_proj_backend_aaa" in trie.lookup("jwt")
        assert "drawer_proj_backend_bbb" in trie.lookup("alembic")
        assert "drawer_proj_frontend_ccc" in trie.lookup("tanstack")
        assert "drawer_notes_planning_ddd" in trie.lookup("passkeys")

    def test_rebuild_is_destructive(self, trie, seeded_collection):
        trie.add_drawer("stale", "stale content", {"filed_at": "2026-01-01"})
        assert "stale" in trie.lookup("stale")

        trie.rebuild_from_collection(seeded_collection)
        # Old stale drawer must be gone after rebuild.
        assert "stale" not in trie.lookup("stale")

    def test_rebuilt_trie_respects_filters(self, trie, seeded_collection):
        trie.rebuild_from_collection(seeded_collection)
        # Wing filter passes through to the rebuilt index
        hits = trie.lookup("jwt", wing="project")
        assert hits == {"drawer_proj_backend_aaa"}


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_lookup_token(self, trie):
        trie.add_drawer("d1", "alpha beta", {"filed_at": "2026-01-01"})
        assert trie.lookup("") == set()

    def test_keyword_search_no_keywords_no_filters(self, trie):
        trie.add_drawer("d1", "alpha", {"filed_at": "2026-01-01"})
        # With nothing to filter on, every posting qualifies.
        hits = trie.keyword_search([])
        assert hits == {"d1"}

    def test_duplicate_add_is_idempotent(self, trie):
        trie.add_drawer("d1", "alpha beta", {"filed_at": "2026-01-01"})
        trie.add_drawer("d1", "alpha beta", {"filed_at": "2026-01-01"})
        stats = trie.stats()
        # Still one drawer, still two tokens ("alpha", "beta").
        assert stats["unique_drawers"] == 1
        assert stats["postings"] == 2

    def test_get_drawer_meta_batching(self, trie):
        # Stress the chunking loop in get_drawer_meta with >400 ids.
        items = []
        for i in range(450):
            items.append(
                (
                    f"d{i}",
                    f"alpha_{i} commonword",
                    {
                        "wing": "w",
                        "room": "r",
                        "filed_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                    },
                )
            )
        trie.add_batch(items)
        meta = trie.get_drawer_meta([f"d{i}" for i in range(450)])
        assert len(meta) == 450
        assert all("filed_at" in m for m in meta.values())


class TestSingletonRegistry:
    """Regression tests for the ``TrieIndex._instances`` registry.

    These cover the test-isolation leak fix added during the Tranche 1
    cleanup: the autouse conftest fixture now clears the registry and
    closes every live instance between tests. Without that, opening
    the same ``db_path`` from two tests would return a stale reference
    to the first test's (closed) env and crash on the next lmdb call.
    """

    def test_close_then_reopen_returns_fresh_instance(self, tmp_dir):
        path = os.path.join(tmp_dir, "reopen_trie.lmdb")
        first = TrieIndex(db_path=path)
        first.add_drawer(
            "d1", "alpha beta gamma", {"wing": "w", "room": "r", "filed_at": "2026-01-01"}
        )
        assert "d1" in first.lookup("alpha")
        first.close()
        # After close, the registry must not return a stale reference.
        assert path not in {
            k for k, v in TrieIndex._instances.items() if getattr(v, "_env", None) is not None
        }

        second = TrieIndex(db_path=path)
        assert second is not first
        # State persists on disk, so the drawer is still visible to the
        # reopened instance.
        assert "d1" in second.lookup("alpha")
        second.close()

    def test_two_instances_at_same_path_share_env(self, tmp_dir):
        path = os.path.join(tmp_dir, "shared_trie.lmdb")
        t1 = TrieIndex(db_path=path)
        t2 = TrieIndex(db_path=path)
        # Must be the same object — py-lmdb forbids two envs at one path.
        assert t1 is t2
        t1.close()


class TestTimezoneAwareDatetimes:
    """Regression test for the Tranche 1 naive→tz-aware datetime
    migration. Every writer now produces ``datetime.now(timezone.utc)``
    timestamps with a ``+00:00`` suffix; the trie's ``_iso_to_days``
    slicer truncates to ``[:19]`` so both naive and tz-aware forms
    map to the same day count and historical data keeps working.
    """

    def test_tz_aware_timestamp_roundtrips(self, trie):
        from datetime import datetime

        tz_now = datetime.now(UTC).isoformat()
        trie.add_drawer(
            "d1",
            "alpha beta gamma",
            {"wing": "w", "room": "r", "filed_at": tz_now},
        )
        meta = trie.get_drawer_meta(["d1"])
        assert "d1" in meta
        # filed_at comes back as a date-only string from the trie
        # (output of `_days_to_iso`); confirm it reflects today.
        assert meta["d1"]["filed_at"].startswith(tz_now[:10])

    def test_mixed_naive_and_tz_timestamps_coexist(self, trie):
        # Legacy (naive) and new (tz-aware) timestamps must produce the
        # same day count — otherwise mixing old and new data would split
        # the same calendar day into two buckets in the time_index.
        from datetime import datetime

        from mempalace.trie_index import _iso_to_days

        naive = datetime(2026, 4, 9, 12, 0, 0).isoformat()
        aware = datetime(2026, 4, 9, 12, 0, 0, tzinfo=UTC).isoformat()
        assert _iso_to_days(naive) == _iso_to_days(aware)


# ── Monotonic temporal fast path + schema migration ────────────────────


class TestMonotonicFastPath:
    """v2 schema additions — the monotonic doc_id temporal shortcut and
    the schema-version drop-on-mismatch migration.
    """

    def test_monotonic_flag_on_sorted_insert(self, trie):
        # 10 drawers in strictly increasing filed_at order should leave
        # the monotonic flag ON.
        for i in range(10):
            trie.add_drawer(
                f"d{i}",
                "alpha beta gamma",
                {
                    "wing": "w",
                    "room": "r",
                    "filed_at": f"2026-01-{i + 1:02d}T00:00:00",
                },
            )
        assert trie.stats()["monotonic"] is True
        assert trie.stats()["mono_checkpoints"] == 10

    def test_monotonic_temporal_fast_path_used(self, trie):
        # Seed with strictly sorted dates and confirm a temporal query
        # takes the mono_fast path.
        for i in range(20):
            trie.add_drawer(
                f"d{i}",
                "jwt token alembic",
                {
                    "wing": "w",
                    "room": "r",
                    "filed_at": f"2026-02-{i + 1:02d}T00:00:00",
                },
            )
        hits = trie.lookup("jwt", since="2026-02-05", until="2026-02-10")
        # filed_at 2026-02-05..2026-02-10 = 6 days = 6 drawers (d4..d9)
        assert len(hits) == 6
        assert trie.stats()["last_query_mode"] == "mono_fast"

    def test_monotonic_breaks_on_out_of_order_insert(self, trie):
        # Insert sorted dates first.
        for i in range(5):
            trie.add_drawer(
                f"d{i}",
                "jwt token",
                {"wing": "w", "room": "r", "filed_at": f"2026-03-{i + 1:02d}T00:00:00"},
            )
        assert trie.stats()["monotonic"] is True

        # Insert an older date → monotonic flag must flip off.
        trie.add_drawer(
            "d_old",
            "jwt token",
            {"wing": "w", "room": "r", "filed_at": "2025-01-01T00:00:00"},
        )
        assert trie.stats()["monotonic"] is False

        # Temporal queries still work, just via the fallback path.
        hits = trie.lookup("jwt", since="2026-03-01", until="2026-03-05")
        assert len(hits) == 5
        assert trie.stats()["last_query_mode"] == "time_index_scan"

    def test_rebuild_from_collection_restores_monotonic(self, trie, seeded_collection):
        # Break monotonicity, then rebuild and confirm it's restored
        # (rebuild_from_collection sorts by filed_at before ingesting).
        trie.add_drawer(
            "first", "jwt", {"wing": "w", "room": "r", "filed_at": "2026-05-01T00:00:00"}
        )
        trie.add_drawer(
            "second", "jwt", {"wing": "w", "room": "r", "filed_at": "2025-01-01T00:00:00"}
        )
        assert trie.stats()["monotonic"] is False

        trie.rebuild_from_collection(seeded_collection)
        # The rebuild sorts by filed_at so monotonicity is restored.
        assert trie.stats()["monotonic"] is True

    def test_schema_migration_drops_old_index(self, tmp_dir):
        # Build a v1-ish index by constructing a v2 trie, then mutating
        # its stored schema version to simulate an old install.
        import os

        from mempalace.trie_index import _META_SCHEMA_VERSION, _U32_LE, TrieIndex

        path = os.path.join(tmp_dir, "legacy_trie.lmdb")
        t1 = TrieIndex(db_path=path)
        t1.add_drawer("legacy", "alembic postgresql", {"wing": "w", "room": "r"})
        assert "legacy" in t1.lookup("alembic")

        # Forge an older schema version in meta.
        with t1._env.begin(write=True) as txn:
            txn.put(_META_SCHEMA_VERSION, _U32_LE.pack(1), db=t1._db_meta)
        t1.close()

        # Reopen — the migration should drop the old index and leave it
        # empty (the Chroma collection is the source of truth).
        t2 = TrieIndex(db_path=path)
        assert t2.stats()["unique_drawers"] == 0
        assert t2.stats()["schema_version"] == 2
        assert t2.lookup("alembic") == set()
        t2.close()

    def test_schema_version_exposed_in_stats(self, trie):
        assert trie.stats()["schema_version"] == 2
        assert trie.stats()["backend"] == "lmdb"

    def test_warm_loads_hot_bitmaps(self, trie):
        # Populate with enough distinct tokens to have ranked cardinalities.
        for i in range(20):
            trie.add_drawer(
                f"d{i}",
                "popular popular popular rare_" + str(i),
                {"wing": "w", "room": "r", "filed_at": "2026-01-01T00:00:00"},
            )
        # Clear the cache to simulate cold state.
        trie._bitmap_cache.clear()
        assert len(trie._bitmap_cache) == 0

        loaded = trie.warm(top_k=5)
        # At least the "popular" token (highest cardinality, 20 drawers)
        # should be in the cache.
        assert loaded >= 1
        assert len(trie._bitmap_cache) >= 1
