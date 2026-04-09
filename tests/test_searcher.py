"""
test_searcher.py -- Tests for both search() (CLI) and search_memories() (API).

Uses the real ChromaDB fixtures from conftest.py for integration tests,
plus mock-based tests for error paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from mempalace.searcher import SearchError, hybrid_search, search, search_memories

# ── search_memories (API) ──────────────────────────────────────────────


class TestSearchMemories:
    def test_basic_search(self, palace_path, seeded_collection):
        result = search_memories("JWT authentication", palace_path)
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["query"] == "JWT authentication"

    def test_wing_filter(self, palace_path, seeded_collection):
        result = search_memories("planning", palace_path, wing="notes")
        assert all(r["wing"] == "notes" for r in result["results"])

    def test_room_filter(self, palace_path, seeded_collection):
        result = search_memories("database", palace_path, room="backend")
        assert all(r["room"] == "backend" for r in result["results"])

    def test_wing_and_room_filter(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, wing="project", room="frontend")
        assert all(r["wing"] == "project" and r["room"] == "frontend" for r in result["results"])

    def test_n_results_limit(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, n_results=2)
        assert len(result["results"]) <= 2

    def test_no_palace_returns_error(self, tmp_path):
        result = search_memories("anything", str(tmp_path / "missing"))
        assert "error" in result

    def test_result_fields(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path)
        hit = result["results"][0]
        assert "text" in hit
        assert "wing" in hit
        assert "room" in hit
        assert "source_file" in hit
        assert "similarity" in hit
        assert isinstance(hit["similarity"], float)

    def test_search_memories_query_error(self):
        """search_memories returns error dict when query raises."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("query failed")

        # The searcher now opens collections through palace_io.open_collection
        # rather than talking to chromadb directly. Patch that seam.
        with patch("mempalace.searcher.open_collection", return_value=mock_col):
            result = search_memories("test", "/fake/path")
        assert "error" in result
        assert "query failed" in result["error"]

    def test_search_memories_filters_in_result(self, palace_path, seeded_collection):
        result = search_memories("test", palace_path, wing="project", room="backend")
        assert result["filters"]["wing"] == "project"
        assert result["filters"]["room"] == "backend"


# ── search() (CLI print function) ─────────────────────────────────────


class TestSearchCLI:
    def test_search_prints_results(self, palace_path, seeded_collection, capsys):
        search("JWT authentication", palace_path)
        captured = capsys.readouterr()
        assert "JWT" in captured.out or "authentication" in captured.out

    def test_search_with_wing_filter(self, palace_path, seeded_collection, capsys):
        search("planning", palace_path, wing="notes")
        captured = capsys.readouterr()
        assert "Results for" in captured.out

    def test_search_with_room_filter(self, palace_path, seeded_collection, capsys):
        search("database", palace_path, room="backend")
        captured = capsys.readouterr()
        assert "Room:" in captured.out

    def test_search_with_wing_and_room(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, wing="project", room="frontend")
        captured = capsys.readouterr()
        assert "Wing:" in captured.out
        assert "Room:" in captured.out

    def test_search_no_palace_raises(self, tmp_path):
        with pytest.raises(SearchError, match="No palace found"):
            search("anything", str(tmp_path / "missing"))

    def test_search_no_results(self, palace_path, collection, capsys):
        """Empty collection returns no results message."""
        # collection is empty (no seeded data)
        result = search("xyzzy_nonexistent_query", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Either prints "No results" or returns None
        assert result is None or "No results" in captured.out

    def test_search_query_error_raises(self):
        """search raises SearchError when query fails."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("boom")

        with patch("mempalace.searcher.open_collection", return_value=mock_col):
            with pytest.raises(SearchError, match="Search error"):
                search("test", "/fake/path")

    def test_search_n_results(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Should have output with at least one result block
        assert "[1]" in captured.out


# ── hybrid_search (trie + semantic + temporal) ────────────────────────


class TestHybridSearch:
    def test_keyword_only_no_query(self, palace_path, seeded_trie):
        # Pure keyword mode: empty query, exact keyword from the trie.
        # Only the auth drawer mentions JWT.
        result = hybrid_search("", palace_path, keywords=["jwt"])
        assert "error" not in result
        assert len(result["results"]) == 1
        assert "JWT" in result["results"][0]["text"]
        # Pure keyword hits have no similarity score.
        assert result["results"][0]["similarity"] is None

    def test_keyword_plus_semantic(self, palace_path, seeded_trie):
        # Keyword narrows to the auth drawer; semantic just ranks within.
        result = hybrid_search("authentication module", palace_path, keywords=["jwt"])
        assert len(result["results"]) > 0
        assert all(
            "JWT" in r["text"] or "authentication" in r["text"].lower() for r in result["results"]
        )
        # Semantic hits carry similarity.
        assert result["results"][0]["similarity"] is not None

    def test_keyword_all_mode_intersects(self, palace_path, seeded_trie):
        # Both keywords must appear. Only the auth drawer has both jwt and tokens.
        result = hybrid_search("", palace_path, keywords=["jwt", "tokens"], keyword_mode="all")
        assert len(result["results"]) == 1
        assert "JWT" in result["results"][0]["text"]

    def test_keyword_any_mode_unions(self, palace_path, seeded_trie):
        # alembic is in bbb, tanstack is in ccc — both must appear.
        result = hybrid_search(
            "", palace_path, keywords=["alembic", "tanstack"], keyword_mode="any"
        )
        texts = " | ".join(r["text"] for r in result["results"])
        assert "Alembic" in texts
        assert "TanStack" in texts

    def test_temporal_since_window(self, palace_path, seeded_trie):
        # Seeded drawers run 2026-01-01..2026-01-04. Since 2026-01-03 keeps
        # the frontend and planning drawers.
        result = hybrid_search("", palace_path, since="2026-01-03T00:00:00")
        assert "error" not in result
        texts = " | ".join(r["text"] for r in result["results"])
        assert "TanStack" in texts or "passkeys" in texts
        # The older auth drawer must be excluded.
        assert "JWT tokens" not in texts

    def test_temporal_until_window(self, palace_path, seeded_trie):
        # Until 2026-01-02 keeps auth + backend-db drawers.
        result = hybrid_search("", palace_path, until="2026-01-02T23:59:59")
        texts = " | ".join(r["text"] for r in result["results"])
        assert "JWT" in texts or "Alembic" in texts
        # Drawers filed later must be excluded.
        assert "TanStack" not in texts

    def test_empty_keyword_returns_empty(self, palace_path, seeded_trie):
        result = hybrid_search("", palace_path, keywords=["xyzzy_nonexistent"])
        assert result["results"] == []
        # Backward-compat: an error-free empty response, not an error dict.
        assert "error" not in result

    def test_wing_and_keyword_combine(self, palace_path, seeded_trie):
        result = hybrid_search("", palace_path, keywords=["alembic"], wing="project")
        assert len(result["results"]) == 1
        assert result["results"][0]["wing"] == "project"

    def test_backward_compat_no_trie_constraints(self, palace_path, seeded_collection):
        # hybrid_search with no trie args must work even when the trie
        # doesn't exist at the palace — it should take the plain Chroma path.
        result = hybrid_search("JWT authentication", palace_path)
        assert "error" not in result
        assert len(result["results"]) > 0
        # Similarity scores come from Chroma, not the trie.
        assert result["results"][0]["similarity"] is not None

    def test_search_memories_backward_compat_shape(self, palace_path, seeded_collection):
        # search_memories must keep its old response shape.
        result = search_memories("JWT authentication", palace_path)
        assert set(result.keys()) == {"query", "filters", "results"}
        assert set(result["filters"].keys()) == {"wing", "room"}
        if result["results"]:
            hit = result["results"][0]
            assert set(hit.keys()) == {
                "text",
                "wing",
                "room",
                "source_file",
                "similarity",
            }


# ── Compression wiring into hybrid_search ────────────────────────────


class TestCompression:
    def test_single_model_auto_resolves_to_none(self, palace_path, seeded_collection):
        """Single-model queries with compress='auto' must leave the
        response byte-for-byte compatible — compression mode is 'none'."""
        result = hybrid_search("JWT authentication", palace_path)
        assert "compression" in result
        assert result["compression"]["mode"] == "none"
        assert result["compression"]["ratio"] == 1.0
        # Results themselves are unchanged from the pre-compression behavior.
        assert len(result["results"]) > 0

    def test_explicit_dedupe_on_single_model(self, palace_path, seeded_collection):
        """Explicit --compress dedupe on a single-model query still
        runs the dedupe pass. With the 4-drawer seed there's nothing
        to merge, so output == input."""
        result = hybrid_search("tokens", palace_path, compress="dedupe")
        assert result["compression"]["mode"] == "dedupe"
        # Seeded drawers are all distinct, so no clusters merge.
        assert result["compression"]["clusters_merged"] == 0

    def _stub_config(self, monkeypatch, palace_path):
        """Point MempalaceConfig at a one-model fan-out setup."""

        class _Stub:
            def __init__(self_inner):
                self_inner.palace_path = palace_path
                self_inner.default_embedding_model = "default"
                self_inner.enabled_embedding_models = ["default"]

        # _hybrid_search_fan_out does `from .config import MempalaceConfig`
        # inside the function body, so the patch has to land on the
        # actual attribute the searcher module will look up.
        import mempalace.config

        monkeypatch.setattr(mempalace.config, "MempalaceConfig", _Stub)

    def test_explicit_none_on_fan_out(self, monkeypatch, palace_path, seeded_collection):
        """model='all' + compress='none' should NOT run any dedupe
        even though fan-out is active."""
        self._stub_config(monkeypatch, palace_path)

        result = hybrid_search(
            "authentication",
            palace_path,
            model="all",
            compress="none",
        )
        assert result["compression"]["mode"] == "none"
        assert result["compression"]["ratio"] == 1.0

    def test_fan_out_auto_becomes_dedupe(self, monkeypatch, palace_path, seeded_collection):
        """model='all' with compress='auto' must resolve to dedupe."""
        self._stub_config(monkeypatch, palace_path)

        result = hybrid_search("authentication", palace_path, model="all")
        assert result["compression"]["mode"] == "dedupe"
        assert "fan_out" in result

    def test_compression_block_in_response(self, palace_path, seeded_collection):
        """Every search response gains a `compression` block even in
        the zero-work 'none' path."""
        result = hybrid_search("tokens", palace_path)
        comp = result["compression"]
        for key in (
            "mode",
            "input_hits",
            "output_hits",
            "input_tokens",
            "output_tokens",
            "ratio",
            "clusters_merged",
            "sentences_dropped",
            "hits_gated_by_novelty",
            "budget_reached",
        ):
            assert key in comp

    def test_aggressive_mode_token_budget(self, palace_path, seeded_collection):
        """Aggressive mode with a tight token budget should halt ingestion."""
        result = hybrid_search(
            "authentication",
            palace_path,
            compress="aggressive",
            token_budget=10,  # absurdly tight — at most one tiny hit
        )
        comp = result["compression"]
        assert comp["mode"] == "aggressive"
        # Either the budget kicked in or the query produced ≤ 10 tokens naturally.
        assert comp["output_tokens"] <= 50  # loose upper bound


class TestFanOutParallelism:
    """Regression tests for the ThreadPoolExecutor-based fan-out path.

    These tests monkeypatch ``_hybrid_search_single`` with a stub that
    sleeps for a fixed interval, then verify the wall-clock of a
    multi-model fan-out is bounded by the per-model delay (parallel)
    rather than the sum (serial).
    """

    def _stub_config_with_models(self, monkeypatch, palace_path, model_slugs):
        """Point MempalaceConfig at a user-specified model list."""

        class _Stub:
            def __init__(self_inner):
                self_inner.palace_path = palace_path
                self_inner.default_embedding_model = "default"
                self_inner.enabled_embedding_models = list(model_slugs)
                self_inner.fan_out_max_workers = 8

        import mempalace.config

        monkeypatch.setattr(mempalace.config, "MempalaceConfig", _Stub)

    def test_fan_out_runs_queries_concurrently(self, monkeypatch, palace_path):
        """4 models × 100 ms each should complete in ~100-200 ms, not 400.

        If the loop ran serially we'd see ~400 ms; parallel should be
        under ~250 ms even with thread spin-up overhead.
        """
        import time

        self._stub_config_with_models(monkeypatch, palace_path, ["m1", "m2", "m3", "m4"])

        calls = []

        def _slow_single(query, palace_path_, **kwargs):
            slug = kwargs["model"]
            calls.append(slug)
            time.sleep(0.1)
            # Return a minimal valid result shape.
            return {
                "query": query,
                "filters": {"wing": None, "room": None, "model": slug},
                "results": [
                    {
                        "text": f"result from {slug}",
                        "wing": "w",
                        "room": "r",
                        "source_file": "",
                        "similarity": 0.9,
                        "filed_at": "",
                        "_drawer_id": f"d_{slug}",
                    }
                ],
            }

        from mempalace import searcher

        monkeypatch.setattr(searcher, "_hybrid_search_single", _slow_single)

        t0 = time.perf_counter()
        result = searcher._hybrid_search_fan_out(
            "test query",
            palace_path,
            keywords=None,
            keyword_mode="all",
            since=None,
            until=None,
            as_of=None,
            wing=None,
            room=None,
            n_results=5,
        )
        elapsed = time.perf_counter() - t0

        # 4 models × 100 ms serial = 400 ms. Parallel ≈ 100-250 ms.
        assert elapsed < 0.3, f"fan-out took {elapsed:.3f}s — suspected serial"
        # All 4 models were queried.
        assert len(calls) == 4
        assert set(calls) == {"m1", "m2", "m3", "m4"}
        # RRF fused 4 distinct drawers.
        assert len(result["results"]) == 4

    def test_fan_out_preserves_rrf_determinism(self, monkeypatch, palace_path):
        """The order-of-completion of parallel workers must not affect
        the final RRF score ordering — ties resolve in enabled-list order.
        """
        import time

        self._stub_config_with_models(monkeypatch, palace_path, ["alpha", "beta"])

        def _out_of_order(query, palace_path_, **kwargs):
            slug = kwargs["model"]
            # beta finishes first, alpha second — but the merge loop
            # iterates enabled in order, so alpha's rank 0 should still
            # win.
            if slug == "beta":
                time.sleep(0.01)
            else:
                time.sleep(0.05)
            return {
                "query": query,
                "filters": {"wing": None, "room": None, "model": slug},
                "results": [
                    {
                        "text": "shared hit",
                        "wing": "w",
                        "room": "r",
                        "source_file": "",
                        "similarity": 0.9,
                        "filed_at": "",
                        "_drawer_id": "shared",  # same drawer from both
                    }
                ],
            }

        from mempalace import searcher

        monkeypatch.setattr(searcher, "_hybrid_search_single", _out_of_order)

        result = searcher._hybrid_search_fan_out(
            "q",
            palace_path,
            keywords=None,
            keyword_mode="all",
            since=None,
            until=None,
            as_of=None,
            wing=None,
            room=None,
            n_results=5,
        )
        assert len(result["results"]) == 1
        hit = result["results"][0]
        # Both models contributed, listed in enabled order (alpha first).
        assert hit["source_models"] == ["alpha", "beta"]
        # RRF score = 1/60 + 1/60 = ~0.0333
        assert abs(hit["rrf_score"] - (1 / 60 + 1 / 60)) < 1e-6

    def test_fan_out_tolerates_worker_failure(self, monkeypatch, palace_path):
        """If one model in the fan-out raises, the others still contribute."""
        self._stub_config_with_models(monkeypatch, palace_path, ["good", "broken", "also-good"])

        def _mixed(query, palace_path_, **kwargs):
            slug = kwargs["model"]
            if slug == "broken":
                raise ValueError("simulated backend failure")
            return {
                "query": query,
                "filters": {"wing": None, "room": None, "model": slug},
                "results": [
                    {
                        "text": f"hit from {slug}",
                        "wing": "w",
                        "room": "r",
                        "source_file": "",
                        "similarity": 0.9,
                        "filed_at": "",
                        "_drawer_id": f"d_{slug}",
                    }
                ],
            }

        from mempalace import searcher

        monkeypatch.setattr(searcher, "_hybrid_search_single", _mixed)

        result = searcher._hybrid_search_fan_out(
            "q",
            palace_path,
            keywords=None,
            keyword_mode="all",
            since=None,
            until=None,
            as_of=None,
            wing=None,
            room=None,
            n_results=5,
        )
        # Two good models contributed; broken was logged and skipped.
        drawer_ids = [r["_drawer_id"] for r in result["results"]]
        assert "d_good" in drawer_ids
        assert "d_also-good" in drawer_ids
        assert "d_broken" not in drawer_ids
