"""
test_kg_extract.py — Unit tests for automatic KG triple extraction.

Covers both paths — heuristic (zero deps) and Ollama (monkeypatched
client so CI never touches a real server) — plus the
``knowledge_graph.add_triple`` dedupe + confidence-voting extension.

No real LLM ever runs; no real network calls ever happen.
"""

import json

import pytest

from mempalace.kg_extract import (
    HeuristicExtractor,
    OllamaExtractor,
    _MissingExtrasError,
    get_extractor,
)

# ── Heuristic extractor ──────────────────────────────────────────────


class TestHeuristicExtractor:
    def test_empty_text_returns_empty(self):
        assert HeuristicExtractor().extract("") == []
        assert HeuristicExtractor().extract("   ") == []

    def test_is_a_pattern(self):
        triples = HeuristicExtractor().extract("Alice is a doctor at Acme.")
        # Should catch both "Alice is a doctor" and "Alice works at Acme" ...
        # actually this sentence only has "is a" so test just that one.
        assert any(
            t.subject == "Alice"
            and t.predicate == "is_a"
            and t.obj.lower() == "doctor at acme"
            or (t.subject == "Alice" and t.predicate == "is_a" and "doctor" in t.obj.lower())
            for t in triples
        )

    def test_works_at_pattern(self):
        triples = HeuristicExtractor().extract("Bob works at Acme Corp.")
        works_at = [t for t in triples if t.predicate == "works_at"]
        assert len(works_at) >= 1
        t = works_at[0]
        assert t.subject == "Bob"
        assert "acme" in t.obj.lower()
        assert t.confidence >= 0.7

    def test_lives_in_pattern(self):
        triples = HeuristicExtractor().extract("Max lives in Berlin.")
        lives = [t for t in triples if t.predicate == "lives_in"]
        assert len(lives) == 1
        assert lives[0].subject == "Max"
        assert lives[0].obj == "Berlin"

    def test_loves_pattern(self):
        triples = HeuristicExtractor().extract("Riley loves chess.")
        loves = [t for t in triples if t.predicate == "loves"]
        assert len(loves) == 1
        assert loves[0].subject == "Riley"
        assert loves[0].obj == "chess"

    def test_entity_hints_boost_confidence(self):
        text = "Priya loves rock climbing."
        no_hints = HeuristicExtractor().extract(text)
        with_hints = HeuristicExtractor().extract(text, entity_hints=["Priya"])

        loves_no = next(t for t in no_hints if t.predicate == "loves")
        loves_with = next(t for t in with_hints if t.predicate == "loves")
        assert loves_with.confidence > loves_no.confidence

    def test_object_stopwords_rejected(self):
        """'Alice is a the' shouldn't extract 'the' as an object."""
        triples = HeuristicExtractor().extract("Alice is a the.")
        # Either no match (pattern doesn't fire) or cleaned object
        # was empty and rejected.
        assert not any(t.obj.lower() in {"the", "a", "an"} for t in triples)

    def test_multiple_sentences(self):
        text = "Alice works at Globex. Bob loves jazz. Charlie lives in Paris."
        triples = HeuristicExtractor().extract(text)
        preds = {t.predicate for t in triples}
        assert "works_at" in preds
        assert "loves" in preds
        assert "lives_in" in preds

    def test_dedupes_same_relation(self):
        """Same (subject, predicate, object) shouldn't appear twice."""
        text = "Alice works at Acme. Alice works at Acme."
        triples = HeuristicExtractor().extract(text)
        works = [t for t in triples if t.predicate == "works_at"]
        assert len(works) == 1

    def test_lowercase_subject_rejected(self):
        """'alice is a doctor' (lowercase) should not match since the
        subject group requires an uppercase first letter."""
        triples = HeuristicExtractor().extract("alice is a doctor")
        assert all(t.subject and t.subject[0].isupper() for t in triples)

    def test_source_drawer_id_propagated(self):
        triples = HeuristicExtractor().extract(
            "Alice works at Acme.",
            source_drawer_id="drawer_12345",
        )
        assert all(t.source_drawer_id == "drawer_12345" for t in triples)

    def test_confidence_in_band(self):
        triples = HeuristicExtractor().extract(
            "Alice is a doctor. Bob works at Acme. Charlie lives in Paris."
        )
        for t in triples:
            assert 0.1 <= t.confidence <= 0.9


# ── Extractor factory ────────────────────────────────────────────────


class TestExtractorFactory:
    def test_heuristic_returns_heuristic(self):
        ex = get_extractor("heuristic")
        assert isinstance(ex, HeuristicExtractor)

    def test_ollama_returns_ollama(self):
        ex = get_extractor("ollama", model="llama3.1:8b")
        assert isinstance(ex, OllamaExtractor)
        assert ex._model == "llama3.1:8b"

    def test_default_is_heuristic(self):
        ex = get_extractor()
        assert isinstance(ex, HeuristicExtractor)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown kg_extract mode"):
            get_extractor("telepathy")


# ── Ollama extractor (with monkeypatched client) ────────────────────


class _FakeOllamaClient:
    """Test double for ``ollama.Client``.

    Real ollama.Client().chat() returns a dict-shaped response like
    ``{"message": {"content": "..."}, "done": True, ...}``. We match
    that shape exactly so the extractor's parsing path hits the
    ``isinstance(response, dict)`` branch.
    """

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.chat_calls = []

    def chat(self, *, model, messages, options=None):
        self.chat_calls.append({"model": model, "messages": messages})
        return {"message": {"content": self._response_text}, "done": True}


class TestOllamaExtractor:
    def test_missing_extras_raises(self, monkeypatch):
        """When ollama is not installed, _ensure raises _MissingExtrasError."""
        import importlib.util

        original_find_spec = importlib.util.find_spec

        def _patched_find(name, *args, **kwargs):
            if name == "ollama":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _patched_find)

        ex = OllamaExtractor()
        with pytest.raises(_MissingExtrasError, match="pip install"):
            ex.extract("Alice works at Acme.")

    def test_parses_valid_json_response(self, monkeypatch):
        """Well-formed JSON from the LLM yields Triple objects."""
        fake_response = json.dumps(
            [
                {
                    "subject": "Alice",
                    "predicate": "works_at",
                    "object": "Acme",
                    "confidence": 0.85,
                    "valid_from": "2024-01-01",
                },
                {
                    "subject": "Bob",
                    "predicate": "loves",
                    "object": "jazz",
                    "confidence": 0.6,
                },
            ]
        )
        ex = OllamaExtractor(model="fake-model")
        ex._client = _FakeOllamaClient(fake_response)

        triples = ex.extract("Alice works at Acme. Bob loves jazz.")
        assert len(triples) == 2
        assert triples[0].subject == "Alice"
        assert triples[0].predicate == "works_at"
        assert triples[0].obj == "Acme"
        assert triples[0].confidence == 0.85
        assert triples[0].valid_from == "2024-01-01"
        assert triples[1].subject == "Bob"
        assert triples[1].valid_from is None

    def test_strips_markdown_code_fences(self, monkeypatch):
        """LLMs often wrap JSON in ```json ... ``` — parser must handle it."""
        fake_response = (
            '```json\n[{"subject": "Alice", "predicate": "is_a", "object": "doctor"}]\n```'
        )
        ex = OllamaExtractor()
        ex._client = _FakeOllamaClient(fake_response)

        triples = ex.extract("Alice is a doctor.")
        assert len(triples) == 1
        assert triples[0].subject == "Alice"

    def test_tolerates_prose_around_json(self, monkeypatch):
        """LLM adds extra prose before/after — parser extracts the array."""
        fake_response = (
            "Sure! Here are the triples I found:\n\n"
            '[{"subject": "Max", "predicate": "loves", "object": "chess"}]\n\n'
            "Let me know if you need more."
        )
        ex = OllamaExtractor()
        ex._client = _FakeOllamaClient(fake_response)

        triples = ex.extract("Max loves chess.")
        assert len(triples) == 1
        assert triples[0].subject == "Max"
        assert triples[0].obj == "chess"

    def test_malformed_json_returns_empty(self, monkeypatch):
        """Totally broken JSON → empty list, not a crash."""
        ex = OllamaExtractor()
        ex._client = _FakeOllamaClient("not json at all {{{")
        assert ex.extract("Alice works at Acme.") == []

    def test_single_object_instead_of_array(self, monkeypatch):
        """Some LLMs return one object when they should return a list."""
        fake_response = json.dumps({"subject": "Alice", "predicate": "is_a", "object": "doctor"})
        ex = OllamaExtractor()
        ex._client = _FakeOllamaClient(fake_response)

        triples = ex.extract("Alice is a doctor.")
        assert len(triples) == 1

    def test_clamps_confidence_to_band(self, monkeypatch):
        """Confidences outside [0.1, 0.9] are clamped."""
        fake_response = json.dumps(
            [
                {"subject": "Alice", "predicate": "is_a", "object": "doctor", "confidence": 1.5},
                {"subject": "Bob", "predicate": "loves", "object": "jazz", "confidence": -0.3},
            ]
        )
        ex = OllamaExtractor()
        ex._client = _FakeOllamaClient(fake_response)

        triples = ex.extract("x")
        assert triples[0].confidence == 0.9
        assert triples[1].confidence == 0.1

    def test_missing_required_fields_skipped(self, monkeypatch):
        """Triples without subject/predicate/object are dropped silently."""
        fake_response = json.dumps(
            [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme"},
                {"subject": "", "predicate": "is_a", "object": "doctor"},
                {"subject": "Bob", "predicate": "loves"},  # no object
            ]
        )
        ex = OllamaExtractor()
        ex._client = _FakeOllamaClient(fake_response)

        triples = ex.extract("x")
        assert len(triples) == 1
        assert triples[0].subject == "Alice"

    def test_connection_error_surfaces_friendly(self, monkeypatch):
        """ConnectionError from the client is translated to a RuntimeError
        with ``ollama serve`` guidance."""

        class _DeadClient:
            def chat(self, *, model, messages, options=None):
                raise ConnectionError("connection refused")

        ex = OllamaExtractor(model="llama3.1:8b")
        ex._client = _DeadClient()

        with pytest.raises(RuntimeError, match="Ollama server unreachable"):
            ex.extract("Alice works at Acme.")

    def test_other_error_returns_empty(self, monkeypatch):
        """Non-connection errors from the client log + return empty
        (miner must keep working)."""

        class _ErroringClient:
            def chat(self, *, model, messages, options=None):
                raise RuntimeError("model out of memory")

        ex = OllamaExtractor()
        ex._client = _ErroringClient()

        # Should not raise — logs + returns empty
        result = ex.extract("Alice works at Acme.")
        assert result == []


# ── KnowledgeGraph dedupe + voting ───────────────────────────────────


class TestKGDedupeAndVoting:
    def test_first_insert_wraps_source_in_json_list(self, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        kg.add_triple("Alice", "works_at", "Acme", source_closet="drawer_001")

        # Read the raw column back
        import sqlite3

        conn = sqlite3.connect(kg.db_path)
        row = conn.execute("SELECT source_closet FROM triples").fetchone()
        conn.close()

        parsed = json.loads(row[0])
        assert parsed == ["drawer_001"]

    def test_repeated_insert_merges_source_closets(self, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        kg.add_triple("Alice", "works_at", "Acme", source_closet="drawer_001")
        kg.add_triple("Alice", "works_at", "Acme", source_closet="drawer_002")
        kg.add_triple("Alice", "works_at", "Acme", source_closet="drawer_003")

        import sqlite3

        conn = sqlite3.connect(kg.db_path)
        rows = conn.execute("SELECT source_closet FROM triples").fetchall()
        conn.close()

        assert len(rows) == 1  # single row, not three
        sources = json.loads(rows[0][0])
        assert sources == ["drawer_001", "drawer_002", "drawer_003"]

    def test_repeated_insert_bumps_confidence(self, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        kg.add_triple("Alice", "works_at", "Acme", confidence=0.5, source_closet="d1")

        import sqlite3

        conn = sqlite3.connect(kg.db_path)
        conf_after_1 = conn.execute("SELECT confidence FROM triples").fetchone()[0]
        conn.close()
        assert abs(conf_after_1 - 0.5) < 0.01

        kg.add_triple("Alice", "works_at", "Acme", confidence=0.5, source_closet="d2")

        conn = sqlite3.connect(kg.db_path)
        conf_after_2 = conn.execute("SELECT confidence FROM triples").fetchone()[0]
        conn.close()
        # 1 - (1 - 0.5) * (1 - 0.5) = 1 - 0.25 = 0.75
        assert abs(conf_after_2 - 0.75) < 0.01

        kg.add_triple("Alice", "works_at", "Acme", confidence=0.5, source_closet="d3")

        conn = sqlite3.connect(kg.db_path)
        conf_after_3 = conn.execute("SELECT confidence FROM triples").fetchone()[0]
        conn.close()
        # 1 - (1 - 0.75) * (1 - 0.5) = 1 - 0.125 = 0.875
        assert abs(conf_after_3 - 0.875) < 0.01

    def test_duplicate_source_closet_not_double_added(self, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        kg.add_triple("Alice", "works_at", "Acme", source_closet="d1")
        kg.add_triple("Alice", "works_at", "Acme", source_closet="d1")  # same!

        import sqlite3

        conn = sqlite3.connect(kg.db_path)
        row = conn.execute("SELECT source_closet FROM triples").fetchone()
        conn.close()
        sources = json.loads(row[0])
        assert sources == ["d1"]  # not [d1, d1]

    def test_no_source_closet_stays_none(self, tmp_path):
        from mempalace.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite3"))
        kg.add_triple("Alice", "works_at", "Acme")

        import sqlite3

        conn = sqlite3.connect(kg.db_path)
        row = conn.execute("SELECT source_closet FROM triples").fetchone()
        conn.close()
        assert row[0] is None


# ── extract_from_palace (engine behind CLI + --extract-kg flag) ─────


class TestExtractFromPalace:
    """Covers the palace walker that wires the extractors to KG writes."""

    def test_cannot_open_collection_reports_error(self, tmp_path, monkeypatch):
        """Open failure is logged and reported as a 1-error stats envelope."""
        from mempalace.kg_extract import extract_from_palace

        def _boom(*args, **kwargs):
            raise RuntimeError("missing palace")

        monkeypatch.setattr("mempalace.palace_io.open_collection", _boom)
        stats = extract_from_palace(str(tmp_path / "nowhere"), mode="heuristic")
        assert stats["drawers_scanned"] == 0
        assert stats["triples_added"] == 0
        assert stats["errors"] == 1
        assert "missing palace" in stats["error"]

    def test_heuristic_walks_single_batch(self, tmp_path, monkeypatch):
        """A single-page collection yields stats consistent with extractor output."""
        from mempalace import kg_extract
        from mempalace.kg_extract import Triple, extract_from_palace

        fake_col = _FakeCollection(
            ids=["d1", "d2"],
            documents=[
                "Alice works at Acme.",
                "Bob lives in Boston.",
            ],
        )
        monkeypatch.setattr("mempalace.palace_io.open_collection", lambda *a, **k: fake_col)

        class _FakeExtractor:
            def extract(self, text, *, source_drawer_id=None):
                return [Triple("X", "rel", "Y", confidence=0.6)]

        monkeypatch.setattr(kg_extract, "get_extractor", lambda *a, **k: _FakeExtractor())

        fake_kg = _FakeKG()
        monkeypatch.setattr(
            "mempalace.knowledge_graph.KnowledgeGraph", lambda *a, **k: fake_kg
        )

        stats = extract_from_palace(str(tmp_path / "palace"), mode="heuristic")
        assert stats["drawers_scanned"] == 2
        assert stats["triples_added"] == 2
        assert stats["errors"] == 0
        assert fake_kg.calls == 2

    def test_extractor_failure_counts_as_error(self, tmp_path, monkeypatch):
        """extractor.extract() exceptions bump the error count but don't halt the walk."""
        from mempalace import kg_extract
        from mempalace.kg_extract import extract_from_palace

        fake_col = _FakeCollection(ids=["d1", "d2"], documents=["text 1", "text 2"])
        monkeypatch.setattr("mempalace.palace_io.open_collection", lambda *a, **k: fake_col)

        class _CrashingExtractor:
            def extract(self, text, *, source_drawer_id=None):
                raise ValueError("boom")

        monkeypatch.setattr(kg_extract, "get_extractor", lambda *a, **k: _CrashingExtractor())
        monkeypatch.setattr(
            "mempalace.knowledge_graph.KnowledgeGraph", lambda *a, **k: _FakeKG()
        )

        stats = extract_from_palace(str(tmp_path / "palace"), mode="heuristic")
        assert stats["drawers_scanned"] == 2
        assert stats["triples_added"] == 0
        assert stats["errors"] == 2

    def test_kg_add_triple_failure_counts_as_error(self, tmp_path, monkeypatch):
        """A failing ``kg.add_triple`` increments the error counter but keeps going."""
        from mempalace import kg_extract
        from mempalace.kg_extract import Triple, extract_from_palace

        fake_col = _FakeCollection(ids=["d1"], documents=["text"])
        monkeypatch.setattr("mempalace.palace_io.open_collection", lambda *a, **k: fake_col)

        class _FakeExtractor:
            def extract(self, text, *, source_drawer_id=None):
                return [Triple("X", "rel", "Y", confidence=0.6)]

        monkeypatch.setattr(kg_extract, "get_extractor", lambda *a, **k: _FakeExtractor())

        class _CrashingKG:
            def add_triple(self, *args, **kwargs):
                raise RuntimeError("db full")

        monkeypatch.setattr(
            "mempalace.knowledge_graph.KnowledgeGraph", lambda *a, **k: _CrashingKG()
        )

        stats = extract_from_palace(str(tmp_path / "palace"), mode="heuristic")
        assert stats["errors"] == 1
        assert stats["triples_added"] == 0

    def test_empty_collection_returns_zero(self, tmp_path, monkeypatch):
        """A collection with zero drawers returns a zero-stats envelope."""
        from mempalace import kg_extract
        from mempalace.kg_extract import extract_from_palace

        fake_col = _FakeCollection(ids=[], documents=[])
        monkeypatch.setattr("mempalace.palace_io.open_collection", lambda *a, **k: fake_col)
        monkeypatch.setattr(kg_extract, "get_extractor", lambda *a, **k: _NoopExtractor())
        monkeypatch.setattr(
            "mempalace.knowledge_graph.KnowledgeGraph", lambda *a, **k: _FakeKG()
        )

        stats = extract_from_palace(str(tmp_path / "palace"), mode="heuristic")
        assert stats["drawers_scanned"] == 0
        assert stats["triples_added"] == 0

    def test_batch_fetch_failure_breaks_loop(self, tmp_path, monkeypatch):
        """An exception from ``collection.get`` bumps errors and breaks the page loop."""
        from mempalace import kg_extract
        from mempalace.kg_extract import extract_from_palace

        class _CrashingCollection:
            def count(self):
                return 10

            def get(self, **kwargs):
                raise RuntimeError("io fail")

        monkeypatch.setattr(
            "mempalace.palace_io.open_collection", lambda *a, **k: _CrashingCollection()
        )
        monkeypatch.setattr(kg_extract, "get_extractor", lambda *a, **k: _NoopExtractor())
        monkeypatch.setattr(
            "mempalace.knowledge_graph.KnowledgeGraph", lambda *a, **k: _FakeKG()
        )

        stats = extract_from_palace(str(tmp_path / "palace"), mode="heuristic")
        assert stats["errors"] == 1
        assert stats["drawers_scanned"] == 0


# Helpers for the extract_from_palace tests above.


class _FakeCollection:
    def __init__(self, ids: list, documents: list):
        self._ids = list(ids)
        self._documents = list(documents)
        self._metadatas = [{} for _ in ids]

    def count(self) -> int:
        return len(self._ids)

    def get(self, *, include=None, limit=500, offset=0):
        end = offset + limit
        return {
            "ids": self._ids[offset:end],
            "documents": self._documents[offset:end],
            "metadatas": self._metadatas[offset:end],
        }


class _FakeKG:
    def __init__(self):
        self.calls = 0

    def add_triple(self, *args, **kwargs):
        self.calls += 1


class _NoopExtractor:
    def extract(self, text, *, source_drawer_id=None):
        return []
