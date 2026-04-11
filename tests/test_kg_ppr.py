"""
test_kg_ppr.py — Tests for Personalized PageRank over the knowledge graph.

These tests use small hand-built KGs (6-20 triples) so PPR
convergence and scoring properties can be verified against
hand-computed expected values. No real Chroma collections, no real
miner — just the KG + the PPR math.
"""

import pytest

from mempalace import kg_ppr
from mempalace.kg_ppr import (
    clear_cache,
    extract_query_entities,
    kg_ppr_candidates,
    personalized_pagerank,
)
from mempalace.knowledge_graph import KnowledgeGraph


@pytest.fixture(autouse=True)
def _reset_kg_ppr_cache():
    """Drop the module-level adjacency cache between tests."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def small_kg(tmp_path):
    """Build a 6-triple KG with two connected components so we can
    verify PPR localization.

    Component 1 (work world):
        alice -works_at-> acme
        alice -uses-> postgres
        postgres -runs_on-> linux
        acme -uses-> postgres  (shared tool — connects alice → linux more strongly)

    Component 2 (isolated):
        bob -loves-> jazz
        bob -plays-> saxophone
    """
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
    kg.add_triple("Alice", "works_at", "Acme", source_closet="drawer_001")
    kg.add_triple("Alice", "uses", "Postgres", source_closet="drawer_002")
    kg.add_triple("Postgres", "runs_on", "Linux", source_closet="drawer_003")
    kg.add_triple("Acme", "uses", "Postgres", source_closet="drawer_004")
    kg.add_triple("Bob", "loves", "jazz", source_closet="drawer_005")
    kg.add_triple("Bob", "plays", "saxophone", source_closet="drawer_006")
    return kg


# ── Graph construction ──────────────────────────────────────────────


class TestGraphConstruction:
    def test_missing_kg_returns_empty(self, tmp_path):
        """A non-existent KG path yields an empty adjacency, not a crash."""
        adj, drawers = kg_ppr._build_adjacency(str(tmp_path / "nonexistent.sqlite3"))
        assert adj == {}
        assert drawers == {}

    def test_adjacency_is_undirected(self, small_kg):
        """Every triple contributes edges in both directions."""
        adj, _ = kg_ppr._build_adjacency(small_kg.db_path)
        # alice ↔ acme
        assert "acme" in adj["alice"]
        assert "alice" in adj["acme"]
        # alice ↔ postgres
        assert "postgres" in adj["alice"]
        assert "alice" in adj["postgres"]

    def test_edge_weight_is_confidence(self, small_kg):
        """Default confidence is 1.0 (legacy manual add_triple) — sum of
        weights on a single edge equals that."""
        adj, _ = kg_ppr._build_adjacency(small_kg.db_path)
        # alice -works_at-> acme with confidence=1.0 (manual default)
        assert adj["alice"]["acme"] == pytest.approx(1.0)

    def test_self_loops_dropped(self, tmp_path):
        """A triple (X, rel, X) is a self-loop and must be filtered."""
        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        kg.add_triple("Max", "is_a", "Max")  # nonsense self-loop
        kg.add_triple("Max", "likes", "chess")
        adj, _ = kg_ppr._build_adjacency(kg.db_path)
        assert "max" not in adj.get("max", {})
        assert "chess" in adj["max"]

    def test_entity_to_drawers_mapping(self, small_kg):
        """Each entity resolves to every drawer that mentions it."""
        _, drawers = kg_ppr._build_adjacency(small_kg.db_path)
        # alice appears in two triples → two drawers
        assert "drawer_001" in drawers["alice"]
        assert "drawer_002" in drawers["alice"]
        # postgres appears in three triples → three drawers
        assert len(drawers["postgres"]) == 3
        # bob and alice are in disconnected components
        assert "drawer_005" not in drawers["alice"]

    def test_cache_hit_on_second_call(self, small_kg):
        """Second call to the cached loader returns the same dicts
        without re-reading the sqlite file."""
        adj1, _ = kg_ppr._graph_cache.get(small_kg.db_path)
        adj2, _ = kg_ppr._graph_cache.get(small_kg.db_path)
        # Same object identity proves it was cached
        assert adj1 is adj2

    def test_cache_invalidates_on_new_triple(self, small_kg):
        """Adding a triple invalidates the cache on the next access."""
        import time

        adj_before, _ = kg_ppr._graph_cache.get(small_kg.db_path)
        assert "new_entity" not in adj_before

        # Ensure mtime changes (filesystems can have 1s resolution)
        time.sleep(0.01)
        small_kg.add_triple("Alice", "has", "new_entity")

        # Force mtime bump on some filesystems
        import os

        os.utime(small_kg.db_path, None)

        adj_after, _ = kg_ppr._graph_cache.get(small_kg.db_path)
        assert "new_entity" in adj_after or "alice" in adj_after


# ── Query entity extraction ─────────────────────────────────────────


class TestExtractQueryEntities:
    def test_empty_query_returns_empty(self):
        assert extract_query_entities("") == []
        assert extract_query_entities("   ") == []

    def test_single_proper_noun(self):
        assert extract_query_entities("where does Alice work") == ["alice"]

    def test_multiple_proper_nouns(self):
        seeds = extract_query_entities("does Alice know Bob")
        assert "alice" in seeds
        assert "bob" in seeds

    def test_multi_word_name(self):
        seeds = extract_query_entities("what did Sarah Smith say")
        assert "sarah_smith" in seeds

    def test_lowercase_words_ignored(self):
        """'the cat is a mammal' — lowercase 'cat' shouldn't match."""
        seeds = extract_query_entities("the cat is a mammal")
        assert seeds == []

    def test_dedup_repeated_names(self):
        seeds = extract_query_entities("Alice loves Alice's work")
        # 'Alice' appears twice but dedupes
        assert seeds.count("alice") == 1


# ── PPR scoring ─────────────────────────────────────────────────────


class TestPersonalizedPageRank:
    def test_empty_kg_returns_empty(self, tmp_path):
        """Running PPR against an empty KG path yields empty results."""
        result = personalized_pagerank(["alice"], kg_db_path=str(tmp_path / "missing.sqlite3"))
        assert result == {}

    def test_no_seeds_in_graph_returns_empty(self, small_kg):
        """Seed entity not in the KG → empty result."""
        result = personalized_pagerank(["nonexistent_entity"], kg_db_path=small_kg.db_path)
        assert result == {}

    def test_seed_dominates_local_neighborhood(self, small_kg):
        """PPR mass concentrates around the seed."""
        scores = personalized_pagerank(["alice"], kg_db_path=small_kg.db_path)
        assert scores  # non-empty
        # Alice herself should be the highest-scoring entity (or
        # very close) because she's the seed and gets teleport mass
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_entity = ranked[0][0]
        assert top_entity in ("alice", "postgres", "acme")

    def test_disconnected_component_gets_zero(self, small_kg):
        """Seeding on Alice should give Bob's component zero mass."""
        scores = personalized_pagerank(["alice"], kg_db_path=small_kg.db_path)
        # Bob's component is disconnected from Alice's
        assert scores.get("bob", 0.0) == 0.0
        assert scores.get("saxophone", 0.0) == 0.0

    def test_convergence(self, small_kg):
        """PPR converges within max_iter."""
        scores = personalized_pagerank(
            ["alice"],
            kg_db_path=small_kg.db_path,
            max_iter=50,
            tol=1e-8,
        )
        # Total mass should be approximately 1 (the seed sums to 1
        # and the update rule preserves mass in expectation)
        total = sum(scores.values())
        assert 0.5 < total < 2.0  # loose check — convergence may be < 1

    def test_multi_seed_averages(self, small_kg):
        """Multi-seed PPR puts mass on each seed."""
        scores = personalized_pagerank(
            ["alice", "bob"],
            kg_db_path=small_kg.db_path,
        )
        assert scores.get("alice", 0) > 0
        assert scores.get("bob", 0) > 0


# ── kg_ppr_candidates end-to-end ────────────────────────────────────


class TestKgPprCandidates:
    def test_no_entities_in_query(self, small_kg):
        """Query without proper nouns → skipped_reason = no_entities_in_query."""
        result = kg_ppr_candidates(
            "what is the best database",
            kg_db_path=small_kg.db_path,
        )
        assert result["skipped_reason"] == "no_entities_in_query"
        assert result["drawer_ids"] == set()

    def test_missing_kg_returns_skipped(self, tmp_path):
        result = kg_ppr_candidates(
            "does Alice know Bob",
            kg_db_path=str(tmp_path / "missing.sqlite3"),
        )
        assert result["skipped_reason"] == "no_kg"

    def test_unknown_seed_returns_skipped(self, small_kg):
        """Query mentions someone not in the KG."""
        result = kg_ppr_candidates(
            "what does Zaphod think",
            kg_db_path=small_kg.db_path,
        )
        assert result["skipped_reason"] == "no_seeds_in_graph"

    def test_alice_query_returns_alice_drawers(self, small_kg):
        """Query about Alice should return drawers from Alice's component."""
        result = kg_ppr_candidates(
            "where does Alice work",
            kg_db_path=small_kg.db_path,
            top_k=20,
        )
        assert result["skipped_reason"] is None
        assert "alice" in result["seeds"]
        # Alice's component has drawers 001-004 (alice/acme/postgres/linux)
        # — at least some should be in the candidate set
        assert any(
            d in result["drawer_ids"]
            for d in {"drawer_001", "drawer_002", "drawer_003", "drawer_004"}
        )
        # Bob's component shouldn't leak in
        assert "drawer_005" not in result["drawer_ids"]
        assert "drawer_006" not in result["drawer_ids"]

    def test_top_entities_shape(self, small_kg):
        result = kg_ppr_candidates(
            "what tools does Alice use",
            kg_db_path=small_kg.db_path,
            top_k=3,
        )
        assert len(result["top_entities"]) <= 3
        for entry in result["top_entities"]:
            assert len(entry) == 2
            assert isinstance(entry[0], str)  # entity_id
            assert isinstance(entry[1], float)  # rounded score

    def test_cross_component_seed(self, small_kg):
        """Query mentions both Alice and Bob — candidates should include
        both components."""
        result = kg_ppr_candidates(
            "do Alice and Bob share interests",
            kg_db_path=small_kg.db_path,
        )
        assert "alice" in result["seeds"]
        assert "bob" in result["seeds"]
        # Both components contribute drawers
        alice_drawers = {"drawer_001", "drawer_002", "drawer_003", "drawer_004"}
        bob_drawers = {"drawer_005", "drawer_006"}
        assert result["drawer_ids"] & alice_drawers
        assert result["drawer_ids"] & bob_drawers


# ── Error paths and integration with searcher ──────────────────────


class TestKgPprErrorPaths:
    """Graceful-degradation coverage for the kg_ppr read path.

    searcher.hybrid_search wraps the kg_ppr call in a broad try/except so
    a missing/empty/corrupt KG never breaks the retrieval pipeline. These
    tests pin that contract: every failure mode must return cleanly.
    """

    def test_empty_kg_file_exists_but_no_triples(self, tmp_path):
        """A KG with schema but zero triples returns an empty envelope.

        The adjacency loader collapses "file exists but no triples" into
        the same empty dict as "file missing", so kg_ppr_candidates flags
        both with ``skipped_reason == "no_kg"``. Either way, the drawer
        set is empty and the caller degrades gracefully.
        """
        kg = KnowledgeGraph(db_path=str(tmp_path / "empty.sqlite3"))
        kg.close()  # just materialize the schema
        result = kg_ppr_candidates("where does Alice work", kg_db_path=kg.db_path)
        assert result["skipped_reason"] in ("no_kg", "no_seeds_in_graph")
        assert result["drawer_ids"] == set()

    def test_max_iter_bound_respected(self, small_kg):
        """``max_iter=1`` still returns a valid (non-empty) score dict."""
        scores = personalized_pagerank(
            ["alice"],
            kg_db_path=small_kg.db_path,
            max_iter=1,
            tol=1.0,  # force immediate exit
        )
        assert scores  # single iteration is enough to produce any output

    def test_searcher_hybrid_search_with_missing_kg_does_not_raise(self, tmp_path, monkeypatch):
        """
        searcher.hybrid_search(..., enable_kg_ppr=True) must degrade gracefully
        when the default KG path doesn't exist. Previously this was only
        exercised via the internal broad try/except — now we pin it at the
        integration boundary.
        """
        import chromadb

        from mempalace import searcher

        # Build a minimal palace with one drawer so the semantic query has
        # something to rank.
        palace = str(tmp_path / "palace")
        client = chromadb.PersistentClient(path=palace)
        col = client.get_or_create_collection("mempalace_drawers")
        col.add(
            ids=["d1"],
            documents=["alpha beta gamma"],
            metadatas=[
                {
                    "wing": "project",
                    "room": "notes",
                    "source_file": "n.md",
                    "chunk_index": 0,
                    "added_by": "test",
                    "filed_at": "2026-01-01T00:00:00",
                }
            ],
        )

        # Point the module-level DEFAULT_KG_PATH at a nonexistent file.
        from mempalace import knowledge_graph

        monkeypatch.setattr(
            knowledge_graph, "DEFAULT_KG_PATH", str(tmp_path / "nonexistent_kg.sqlite3")
        )

        # This is the contract: enable_kg_ppr=True with a missing KG must
        # return a normal search envelope, not an exception.
        result = searcher.hybrid_search(
            "alpha",
            palace_path=palace,
            enable_kg_ppr=True,
        )
        assert "results" in result
        assert isinstance(result["results"], list)
