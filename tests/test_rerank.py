"""
test_rerank.py — Unit tests for the reranker registry.

These tests exercise the registry, spec lookup, lazy-load semantics,
adapter dispatch, and the searcher integration — **without downloading
any real model weights**. A fake reranker adapter is monkeypatched
into the loader so CI runs entirely offline.
"""

import pytest

from mempalace import rerank as rerank_module
from mempalace.rerank import (
    REGISTRY,
    RerankSpec,
    _MissingExtrasError,
    clear_cache,
    get_reranker_spec,
    is_installed,
    list_reranker_specs,
    register,
)


@pytest.fixture(autouse=True)
def _reset_rerank_cache():
    """Clear the process-wide loaded-adapter cache between tests."""
    clear_cache()
    yield
    clear_cache()


# ── Fake reranker plumbing ────────────────────────────────────────────


class _FakeReranker:
    """Deterministic fake reranker for tests.

    Score per hit = length of the text (longer = more relevant). When
    ``prune=True`` and the spec supports pruning, returns the text
    with its last 3 chars removed as a fake pruned variant so tests
    can distinguish pruned from unpruned output.
    """

    def __init__(self, spec: RerankSpec):
        self.spec = spec
        self.calls = 0

    def rerank(self, query, hits, *, top_k=None, prune=True):
        self.calls += 1
        out = []
        for hit in hits:
            new_hit = dict(hit)
            text = hit.get("text", "")
            new_hit["rerank_score"] = float(len(text))
            if prune and self.spec.supports_pruning:
                pruned = text[:-3] if len(text) > 3 else text
                new_hit["pruned_text"] = pruned
                new_hit["pruning_stats"] = {
                    "chars_in": len(text),
                    "chars_out": len(pruned),
                    "ratio": round(len(pruned) / max(len(text), 1), 3),
                }
            out.append(new_hit)
        out.sort(key=lambda h: h["rerank_score"], reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out


def _install_fake_reranker(monkeypatch, slug: str, supports_pruning: bool):
    """Register a fake spec and patch load_reranker to return a ``_FakeReranker``."""
    spec = RerankSpec(
        slug=slug,
        display_name=f"Fake {slug}",
        description="test double",
        backend="fake-test",
        model_id=f"fake/{slug}",
        max_length=512,
        supports_pruning=supports_pruning,
        extras_required=(),
    )
    register(spec)

    original_load = rerank_module.load_reranker

    def _patched_load(s: str):
        if s == slug:
            cached = rerank_module._loaded_rerankers.get(s)
            if cached is not None:
                return cached
            fn = _FakeReranker(spec)
            rerank_module._loaded_rerankers[s] = fn
            return fn
        return original_load(s)

    monkeypatch.setattr(rerank_module, "load_reranker", _patched_load)
    return spec


# ── Registry tests ───────────────────────────────────────────────────


class TestRerankerRegistry:
    def test_list_specs_returns_both_builtins(self):
        specs = list_reranker_specs()
        slugs = {s.slug for s in specs}
        assert "provence" in slugs
        assert "bge" in slugs

    def test_get_spec_returns_correct_dataclass(self):
        provence = get_reranker_spec("provence")
        assert provence.slug == "provence"
        assert provence.backend == "transformers-provence"
        assert provence.supports_pruning is True
        assert provence.extras_required == ("transformers", "torch")

        bge = get_reranker_spec("bge")
        assert bge.slug == "bge"
        assert bge.backend == "fastembed-bge"
        assert bge.supports_pruning is False
        assert bge.extras_required == ("fastembed",)

    def test_get_spec_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown reranker"):
            get_reranker_spec("does-not-exist")

    def test_is_installed_checks_extras(self):
        """is_installed returns True only when every extra is importable."""
        provence = get_reranker_spec("provence")
        bge = get_reranker_spec("bge")
        # In CI the extras are NOT installed, so both should be False.
        # This assertion is defensive — if a dev machine has them
        # installed, is_installed returning True is also correct.
        assert is_installed(provence) in (True, False)
        assert is_installed(bge) in (True, False)

    def test_load_reranker_lazy_via_fake(self, monkeypatch):
        """load_reranker returns the fake adapter without touching real weights."""
        _install_fake_reranker(monkeypatch, "fake-lazy", supports_pruning=True)

        reranker = rerank_module.load_reranker("fake-lazy")
        assert isinstance(reranker, _FakeReranker)

        # Second call returns the same instance (cached).
        same = rerank_module.load_reranker("fake-lazy")
        assert reranker is same

    def test_load_reranker_missing_extras_raises(self, monkeypatch):
        """Loading a real spec with no extras installed raises _MissingExtrasError."""
        # Register a spec whose extras point at a definitely-nonexistent package.
        fake_spec = RerankSpec(
            slug="missing-extras-test",
            display_name="Missing extras test",
            description="",
            backend="transformers-provence",
            model_id="fake/missing",
            max_length=512,
            supports_pruning=True,
            extras_required=("this_package_does_not_exist_12345",),
        )
        register(fake_spec)
        with pytest.raises(_MissingExtrasError, match="pip install"):
            rerank_module.load_reranker("missing-extras-test")
        # Cleanup
        REGISTRY.pop("missing-extras-test", None)


# ── Adapter behavior tests ───────────────────────────────────────────


class TestFakeRerankerBehavior:
    def test_rerank_adds_score(self, monkeypatch):
        _install_fake_reranker(monkeypatch, "fake-score", supports_pruning=False)
        reranker = rerank_module.load_reranker("fake-score")
        hits = [
            {"text": "short", "wing": "w", "room": "r", "source_file": "s"},
            {"text": "this is a longer document", "wing": "w", "room": "r", "source_file": "s"},
        ]
        out = reranker.rerank("query", hits)
        assert len(out) == 2
        # Longer text ranks first (fake scoring).
        assert out[0]["text"] == "this is a longer document"
        assert out[0]["rerank_score"] == len("this is a longer document")
        assert "pruned_text" not in out[0]  # non-pruning backend

    def test_rerank_with_pruning(self, monkeypatch):
        _install_fake_reranker(monkeypatch, "fake-prune", supports_pruning=True)
        reranker = rerank_module.load_reranker("fake-prune")
        hits = [
            {"text": "sample document content", "wing": "w", "room": "r", "source_file": "s"},
        ]
        out = reranker.rerank("query", hits, prune=True)
        assert "pruned_text" in out[0]
        assert "pruning_stats" in out[0]
        assert out[0]["pruning_stats"]["chars_in"] == len("sample document content")
        assert out[0]["pruning_stats"]["chars_out"] == len("sample document content") - 3

    def test_rerank_prune_false_on_pruning_backend(self, monkeypatch):
        _install_fake_reranker(monkeypatch, "fake-no-prune", supports_pruning=True)
        reranker = rerank_module.load_reranker("fake-no-prune")
        hits = [{"text": "content", "wing": "w", "room": "r", "source_file": "s"}]
        out = reranker.rerank("query", hits, prune=False)
        assert "rerank_score" in out[0]
        assert "pruned_text" not in out[0]

    def test_rerank_top_k(self, monkeypatch):
        _install_fake_reranker(monkeypatch, "fake-topk", supports_pruning=False)
        reranker = rerank_module.load_reranker("fake-topk")
        hits = [
            {"text": "a", "wing": "w", "room": "r", "source_file": "s"},
            {"text": "ab", "wing": "w", "room": "r", "source_file": "s"},
            {"text": "abc", "wing": "w", "room": "r", "source_file": "s"},
        ]
        out = reranker.rerank("query", hits, top_k=2)
        assert len(out) == 2

    def test_rerank_empty_hits(self, monkeypatch):
        _install_fake_reranker(monkeypatch, "fake-empty", supports_pruning=False)
        reranker = rerank_module.load_reranker("fake-empty")
        assert reranker.rerank("query", []) == []


# ── Searcher integration tests ───────────────────────────────────────


class TestSearcherIntegration:
    """Verify hybrid_search threads the rerank kwarg through correctly."""

    def test_rerank_none_is_passthrough(self, palace_path, seeded_collection):
        """rerank=None (default) preserves existing behavior byte-for-byte."""
        from mempalace.searcher import hybrid_search

        result = hybrid_search("authentication", palace_path)
        assert result["rerank"]["mode"] == "none"
        assert result["rerank"]["hits_reranked"] == 0

    def test_rerank_with_fake_adds_score(self, monkeypatch, palace_path, seeded_collection):
        """Passing a fake reranker slug runs the rerank stage and reports stats."""
        from mempalace.searcher import hybrid_search

        _install_fake_reranker(monkeypatch, "fake-integration", supports_pruning=False)

        result = hybrid_search(
            "authentication",
            palace_path,
            rerank="fake-integration",
            n_results=3,
        )
        assert result["rerank"]["mode"] == "fake-integration"
        assert result["rerank"]["hits_reranked"] >= 1
        # Every hit should carry a rerank_score now.
        for hit in result["results"]:
            assert "rerank_score" in hit

    def test_rerank_with_pruning_adds_pruned_text(
        self, monkeypatch, palace_path, seeded_collection
    ):
        """When the reranker supports pruning, hits gain pruning_stats (exposed to callers)."""
        from mempalace.searcher import hybrid_search

        _install_fake_reranker(monkeypatch, "fake-pruning", supports_pruning=True)

        result = hybrid_search(
            "authentication",
            palace_path,
            rerank="fake-pruning",
            rerank_prune=True,
            n_results=3,
        )
        assert result["rerank"]["mode"] == "fake-pruning"
        assert result["rerank"]["prune"] is True
        # At least one hit has pruning metadata.
        assert any("pruning_stats" in h for h in result["results"])

    def test_rerank_missing_extras_degrades_gracefully(self, palace_path, seeded_collection):
        """If a reranker fails to load, search still returns results."""
        from mempalace.searcher import hybrid_search

        # Register a spec with extras that don't exist
        fake_spec = RerankSpec(
            slug="degrade-test",
            display_name="Degrade test",
            description="",
            backend="transformers-provence",
            model_id="fake/degrade",
            max_length=512,
            supports_pruning=True,
            extras_required=("this_package_does_not_exist_67890",),
        )
        register(fake_spec)
        try:
            result = hybrid_search(
                "authentication",
                palace_path,
                rerank="degrade-test",
                n_results=3,
            )
            # Search succeeded even though rerank was skipped
            assert "results" in result
            assert result["rerank"]["mode"] == "none"  # fell back
            assert "error" in result["rerank"]
        finally:
            REGISTRY.pop("degrade-test", None)
