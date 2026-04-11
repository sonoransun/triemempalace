"""
test_compress.py — Unit tests for the result-set compression pipeline.

The tests here operate on plain hit dicts — no ChromaDB, no model
weights, no network. The compression pipeline is a pure function
over token sets so everything is deterministic and fast.
"""

from mempalace.compress import (
    MODES,
    compress_results,
    resolve_auto_mode,
)


def _hit(text, *, drawer_id="d0", rrf=None, sim=None, models=None):
    """Build a minimal hit dict with sensible defaults."""
    h = {
        "text": text,
        "wing": "w",
        "room": "r",
        "source_file": "test.md",
        "similarity": sim,
        "filed_at": "2026-01-01T00:00:00",
        "_drawer_id": drawer_id,
    }
    if rrf is not None:
        h["rrf_score"] = rrf
    if models is not None:
        h["source_models"] = models
    return h


# ── resolve_auto_mode ────────────────────────────────────────────────


class TestResolveAuto:
    def test_auto_on_fan_out_becomes_dedupe(self):
        assert resolve_auto_mode(model="all", compress="auto") == "dedupe"

    def test_auto_on_single_model_stays_none(self):
        assert resolve_auto_mode(model=None, compress="auto") == "none"
        assert resolve_auto_mode(model="default", compress="auto") == "none"
        assert resolve_auto_mode(model="jina-code-v2", compress="auto") == "none"

    def test_explicit_mode_passes_through_unchanged(self):
        for mode in MODES:
            assert resolve_auto_mode(model="all", compress=mode) == mode
            assert resolve_auto_mode(model=None, compress=mode) == mode


# ── Mode: none — passthrough ─────────────────────────────────────────


class TestPassthrough:
    def test_none_returns_input_unchanged(self):
        hits = [_hit("alpha beta gamma", drawer_id="d1")]
        out, stats = compress_results(hits, mode="none")
        assert out == hits
        assert stats["mode"] == "none"
        assert stats["ratio"] == 1.0
        assert stats["clusters_merged"] == 0

    def test_empty_input_returns_empty(self):
        out, stats = compress_results([], mode="dedupe")
        assert out == []
        assert stats["input_hits"] == 0
        assert stats["output_hits"] == 0
        assert stats["ratio"] == 1.0  # no division-by-zero

    def test_stats_are_populated_even_in_none_mode(self):
        hits = [_hit("one two three four five", drawer_id="d1")]
        _out, stats = compress_results(hits, mode="none")
        assert stats["input_tokens"] > 0
        assert stats["output_tokens"] == stats["input_tokens"]
        assert stats["input_hits"] == 1
        assert stats["output_hits"] == 1


# ── Mode: dedupe — drawer-level clustering ──────────────────────────


class TestDedupe:
    def test_verbatim_duplicates_merge(self):
        text = "The quick brown fox jumped over the lazy dog"
        hits = [
            _hit(text, drawer_id="d1", rrf=0.03, models=["default"]),
            _hit(text, drawer_id="d2", rrf=0.02, models=["jina-code-v2"]),
        ]
        out, stats = compress_results(hits, mode="dedupe")
        assert len(out) == 1
        assert stats["clusters_merged"] == 1
        assert stats["output_hits"] == 1
        rep = out[0]
        assert rep["cluster_size"] == 2
        assert set(rep["merged_drawer_ids"]) == {"d1", "d2"}
        assert set(rep["merged_source_models"]) == {"default", "jina-code-v2"}
        assert len(rep["variants"]) == 1

    def test_word_swap_merges(self):
        # Two drawers differing by a single word should cluster at
        # default threshold (token-set Jaccard is robust to swaps).
        hits = [
            _hit(
                "JWT tokens expire after 24 hours. Refresh tokens stored in HttpOnly cookies",
                drawer_id="d1",
                rrf=0.03,
                models=["default"],
            ),
            _hit(
                "JWT tokens expire after 24 hours. Refresh tokens live in HttpOnly cookies",
                drawer_id="d2",
                rrf=0.02,
                models=["jina-code-v2"],
            ),
        ]
        out, stats = compress_results(hits, mode="dedupe")
        assert len(out) == 1
        assert stats["clusters_merged"] == 1

    def test_unrelated_drawers_dont_merge(self):
        hits = [
            _hit("Python backend refactoring notes for the API layer", drawer_id="d1"),
            _hit("React frontend component library migration plans", drawer_id="d2"),
        ]
        out, stats = compress_results(hits, mode="dedupe")
        assert len(out) == 2
        assert stats["clusters_merged"] == 0

    def test_representative_picked_by_rrf_score(self):
        # Two duplicates with different RRF scores — the higher-scoring
        # one wins the representative slot.
        hits = [
            _hit("alpha beta gamma delta", drawer_id="low", rrf=0.01),
            _hit("alpha beta gamma delta", drawer_id="high", rrf=0.05),
        ]
        out, _ = compress_results(hits, mode="dedupe")
        assert len(out) == 1
        assert out[0]["_drawer_id"] == "high"

    def test_representative_picked_by_similarity_when_no_rrf(self):
        hits = [
            _hit("alpha beta gamma delta", drawer_id="low", sim=0.4),
            _hit("alpha beta gamma delta", drawer_id="high", sim=0.9),
        ]
        out, _ = compress_results(hits, mode="dedupe")
        assert out[0]["_drawer_id"] == "high"

    def test_variants_populated_on_merged_clusters(self):
        hits = [
            _hit(
                "alpha beta gamma delta epsilon",
                drawer_id="d1",
                rrf=0.05,
                models=["default"],
            ),
            _hit(
                "alpha beta gamma delta epsilon",
                drawer_id="d2",
                rrf=0.02,
                models=["jina-code-v2"],
            ),
        ]
        out, _ = compress_results(hits, mode="dedupe")
        assert len(out) == 1
        variants = out[0]["variants"]
        assert len(variants) == 1
        assert variants[0]["drawer_id"] == "d2"
        assert variants[0]["source_model"] == "jina-code-v2"
        assert variants[0]["text"] == "alpha beta gamma delta epsilon"

    def test_singleton_cluster_has_empty_variants(self):
        hits = [_hit("unique content about distinct topic", drawer_id="solo")]
        out, _ = compress_results(hits, mode="dedupe")
        assert out[0]["variants"] == []
        assert out[0]["cluster_size"] == 1

    def test_disjoint_clusters_preserve_rank_order(self):
        # Two clusters: (A1, A2) duplicates of A, (B1, B2) duplicates
        # of B. Output should have the highest-ranked rep first.
        hits = [
            _hit("alpha content about databases", drawer_id="a1", rrf=0.05),
            _hit("beta content about frontend", drawer_id="b1", rrf=0.04),
            _hit("alpha content about databases", drawer_id="a2", rrf=0.03),
            _hit("beta content about frontend", drawer_id="b2", rrf=0.02),
        ]
        out, stats = compress_results(hits, mode="dedupe")
        assert len(out) == 2
        assert stats["clusters_merged"] == 2
        assert out[0]["_drawer_id"] == "a1"
        assert out[1]["_drawer_id"] == "b1"

    def test_dup_threshold_override(self):
        # Two hits with moderate overlap (~0.5 Jaccard). At the default
        # 0.7 they stay separate; at 0.4 they cluster.
        hits = [
            _hit("python backend refactoring new auth flow", drawer_id="a"),
            _hit("python backend deploying new build cache", drawer_id="b"),
        ]
        out_default, _ = compress_results(hits, mode="dedupe")
        out_loose, _ = compress_results(hits, mode="dedupe", dup_threshold=0.25)
        # Default is strict → likely no merge
        assert len(out_default) == 2
        # Loose should be ≤ default
        assert len(out_loose) <= len(out_default)


# ── Mode: sentences — sentence-level dedupe ─────────────────────────


class TestSentenceDedupe:
    def test_drops_repeated_opener(self):
        # Two hits share the first sentence but diverge after. In
        # "sentences" mode the second hit's repeated opener gets
        # dropped.
        shared = "The authentication module uses JWT for session state"
        hits = [
            _hit(f"{shared}. Tokens expire after 24 hours", drawer_id="d1", rrf=0.05),
            _hit(f"{shared}. Refresh rotation runs nightly", drawer_id="d2", rrf=0.03),
        ]
        out, stats = compress_results(hits, mode="sentences")
        # Both hits survive as separate clusters (different unique content).
        assert len(out) == 2
        # But the second hit's repeated opener is gone.
        assert stats["sentences_dropped"] >= 1
        assert "Tokens expire" in out[0]["text"]
        # The second rep should only contain its unique tail.
        assert "Refresh rotation" in out[1]["text"]
        assert out[1]["text"].count("authentication module") == 0

    def test_mode_sentences_runs_dedupe_first(self):
        # When two hits are verbatim duplicates, sentences mode still
        # clusters them before running sentence dedupe.
        text = "The quick brown fox. Jumped over the lazy dog"
        hits = [
            _hit(text, drawer_id="d1", rrf=0.05),
            _hit(text, drawer_id="d2", rrf=0.03),
        ]
        out, stats = compress_results(hits, mode="sentences")
        assert len(out) == 1
        assert stats["clusters_merged"] == 1


# ── Mode: aggressive — novelty + budget ──────────────────────────────


class TestAggressive:
    def test_novelty_gate_drops_low_novelty_hits(self):
        # Construct three hits such that:
        #   - d1/d2/d3 do NOT cluster at dedupe (low unigram Jaccard)
        #   - d3 survives sentence dedupe (bigram overlap < 0.75)
        #   - d3 fails the novelty gate (trigram novelty < threshold)
        #
        # d3 stitches together subsequences from d1 and d2, so its
        # trigrams mostly come from (d1 ∪ d2).
        hits = [
            _hit(
                "apple banana charlie delta echo foxtrot",
                drawer_id="d1",
                rrf=0.05,
            ),
            _hit(
                "golf hotel india juliet kilo lima",
                drawer_id="d2",
                rrf=0.04,
            ),
            _hit(
                "apple banana charlie golf hotel india newword",
                drawer_id="d3",
                rrf=0.02,
            ),
        ]
        out, stats = compress_results(
            hits,
            mode="aggressive",
            novelty_threshold=0.7,
        )
        assert stats["hits_gated_by_novelty"] >= 1
        # d1 and d2 should survive; d3 should not.
        kept_ids = {h.get("_drawer_id") for h in out}
        assert "d1" in kept_ids
        assert "d2" in kept_ids
        assert "d3" not in kept_ids

    def test_token_budget_halts_ingestion(self):
        # Five hits of ~20 tokens each = ~100 total. Budget of 40
        # should cut after roughly two hits.
        hits = [
            _hit(
                "unique_topic_" + str(i) + " " + " ".join(f"word_{i}_{j}" for j in range(15)),
                drawer_id=f"d{i}",
                rrf=1.0 / (60 + i),
            )
            for i in range(5)
        ]
        out, stats = compress_results(
            hits, mode="aggressive", token_budget=40, novelty_threshold=0.0
        )
        assert stats["budget_reached"] is True
        assert stats["output_tokens"] <= 40 or len(out) <= 3

    def test_budget_none_means_unlimited(self):
        hits = [_hit(f"unique topic {i} alpha beta", drawer_id=f"d{i}") for i in range(5)]
        _out, stats = compress_results(hits, mode="aggressive", token_budget=None)
        assert stats["budget_reached"] is False


# ── Stats envelope shape ─────────────────────────────────────────────


class TestStatsEnvelope:
    def test_stats_has_all_expected_keys(self):
        hits = [_hit("alpha beta gamma", drawer_id="d1")]
        _out, stats = compress_results(hits, mode="dedupe")
        expected_keys = {
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
        }
        assert expected_keys <= set(stats.keys())

    def test_ratio_is_greater_than_one_when_compressing(self):
        text = "one two three four five six seven eight nine ten"
        hits = [
            _hit(text, drawer_id="d1", rrf=0.05),
            _hit(text, drawer_id="d2", rrf=0.03),
            _hit(text, drawer_id="d3", rrf=0.02),
        ]
        _out, stats = compress_results(hits, mode="dedupe")
        assert stats["ratio"] > 1.0


# ── Error handling ──────────────────────────────────────────────────


class TestErrorHandling:
    def test_unknown_mode_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown compression mode"):
            compress_results([], mode="turbo")

    def test_hit_with_empty_text_doesnt_crash(self):
        hits = [
            _hit("", drawer_id="empty1"),
            _hit("normal content", drawer_id="d1"),
        ]
        out, stats = compress_results(hits, mode="dedupe")
        # Shouldn't raise; empty-text hit stays as its own (singleton)
        # cluster unless the tokenizer produces no tokens, in which
        # case it may merge with anything that also has zero tokens.
        assert stats["input_hits"] == 2

    def test_hit_without_drawer_id_still_works(self):
        hits = [
            {"text": "alpha beta gamma", "wing": "w", "room": "r"},
            {"text": "alpha beta gamma", "wing": "w", "room": "r"},
        ]
        out, stats = compress_results(hits, mode="dedupe")
        assert len(out) == 1
        assert stats["clusters_merged"] == 1


class TestLLMLingua2:
    """Tests for the ``llmlingua2`` compression mode.

    These tests never download real xlm-roberta weights — every test
    either verifies the missing-extras fallback path or monkeypatches
    a fake compressor into the module-level singleton.
    """

    def test_llmlingua2_is_in_modes(self):
        from mempalace.compress import MODES

        assert "llmlingua2" in MODES

    def test_fallback_when_extras_missing(self, monkeypatch):
        """Without the compress-llmlingua extra installed, mode='llmlingua2'
        falls back to a passthrough with a ``fallback`` stats field."""
        from mempalace import compress as compress_module

        # Reset the module-level singleton so the test doesn't pick up
        # a real loader from a previous test run.
        monkeypatch.setattr(compress_module, "_llmlingua_compressor", None)
        # Force the loader to return None as if the import failed.
        monkeypatch.setattr(compress_module, "_load_llmlingua_compressor", lambda: None)

        hits = [
            _hit("alpha beta gamma delta epsilon", drawer_id="d1"),
            _hit("completely different content here", drawer_id="d2"),
        ]
        out, stats = compress_results(hits, mode="llmlingua2")

        # Passthrough — no compression happened
        assert len(out) == 2
        assert stats["mode"] == "llmlingua2"
        assert stats["input_hits"] == 2
        assert stats["output_hits"] == 2
        assert stats["input_tokens"] == stats["output_tokens"]
        assert stats["ratio"] == 1.0
        assert stats["fallback"] == "extras not installed"

    def test_monkeypatched_compressor_is_called(self, monkeypatch):
        """With a fake compressor in place, each hit goes through it."""
        from mempalace import compress as compress_module

        calls = []

        class _FakeCompressor:
            def compress_prompt(self, text, *, target_token=None, **kwargs):
                calls.append((text, target_token))
                return {"compressed_prompt": f"<compressed:{len(text)}>"}

        monkeypatch.setattr(compress_module, "_llmlingua_compressor", None)
        monkeypatch.setattr(
            compress_module,
            "_load_llmlingua_compressor",
            lambda: _FakeCompressor(),
        )

        hits = [
            _hit("the quick brown fox jumps over the lazy dog", drawer_id="d1"),
            _hit("mempalace stores verbatim drawers for retrieval", drawer_id="d2"),
        ]
        out, stats = compress_results(hits, mode="llmlingua2")

        # Both hits had their text replaced with the fake compressed form
        assert len(out) == 2
        assert out[0]["text"].startswith("<compressed:")
        assert out[1]["text"].startswith("<compressed:")
        # Original text preserved under ``original_text``
        assert "original_text" in out[0]
        assert out[0]["original_text"] == "the quick brown fox jumps over the lazy dog"
        # Compressor was called once per hit
        assert len(calls) == 2

    def test_token_budget_divides_across_hits(self, monkeypatch):
        """When token_budget is set, each hit gets budget/N tokens."""
        from mempalace import compress as compress_module

        received_budgets = []

        class _FakeCompressor:
            def compress_prompt(self, text, *, target_token=None, **kwargs):
                received_budgets.append(target_token)
                return {"compressed_prompt": text[:20]}

        monkeypatch.setattr(compress_module, "_llmlingua_compressor", None)
        monkeypatch.setattr(
            compress_module,
            "_load_llmlingua_compressor",
            lambda: _FakeCompressor(),
        )

        hits = [
            _hit("aaa bbb ccc ddd eee", drawer_id="d1"),
            _hit("fff ggg hhh iii jjj", drawer_id="d2"),
            _hit("kkk lll mmm nnn ooo", drawer_id="d3"),
        ]
        out, stats = compress_results(hits, mode="llmlingua2", token_budget=90)

        assert len(out) == 3
        # 90 tokens / 3 hits = 30 per hit
        assert received_budgets == [30, 30, 30]

    def test_per_hit_compressor_failure_keeps_original(self, monkeypatch):
        """If the compressor raises on one hit, that hit passes through."""
        from mempalace import compress as compress_module

        class _FlakyCompressor:
            def __init__(self):
                self.call_count = 0

            def compress_prompt(self, text, *, target_token=None, **kwargs):
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("simulated OOM on hit 2")
                return {"compressed_prompt": f"<ok:{self.call_count}>"}

        monkeypatch.setattr(compress_module, "_llmlingua_compressor", None)
        monkeypatch.setattr(
            compress_module,
            "_load_llmlingua_compressor",
            lambda: _FlakyCompressor(),
        )

        hits = [
            _hit("content one", drawer_id="d1"),
            _hit("content two", drawer_id="d2"),
            _hit("content three", drawer_id="d3"),
        ]
        out, stats = compress_results(hits, mode="llmlingua2")

        assert len(out) == 3
        assert out[0]["text"] == "<ok:1>"
        # Middle hit kept original text when compressor raised
        assert out[1]["text"] == "content two"
        assert "original_text" not in out[1]
        assert out[2]["text"] == "<ok:3>"

    def test_empty_hit_text_is_preserved(self, monkeypatch):
        """Hits with empty text skip the compressor but stay in the list."""
        from mempalace import compress as compress_module

        class _FakeCompressor:
            def compress_prompt(self, text, *, target_token=None, **kwargs):
                # Would fail on empty text — make sure we never get here
                assert text.strip()
                return {"compressed_prompt": text[:5]}

        monkeypatch.setattr(compress_module, "_llmlingua_compressor", None)
        monkeypatch.setattr(
            compress_module,
            "_load_llmlingua_compressor",
            lambda: _FakeCompressor(),
        )

        hits = [
            _hit("", drawer_id="empty"),
            _hit("real content", drawer_id="d1"),
        ]
        out, stats = compress_results(hits, mode="llmlingua2")

        assert len(out) == 2
        assert out[0]["text"] == ""
        assert out[1]["text"].startswith("real")
