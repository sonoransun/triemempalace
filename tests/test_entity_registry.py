"""Tests for mempalace.entity_registry."""

from mempalace.entity_registry import (
    COMMON_ENGLISH_WORDS,
    EntityRegistry,
)

# ── COMMON_ENGLISH_WORDS ────────────────────────────────────────────────


def test_common_english_words_has_expected_entries():
    assert "ever" in COMMON_ENGLISH_WORDS
    assert "grace" in COMMON_ENGLISH_WORDS
    assert "will" in COMMON_ENGLISH_WORDS
    assert "may" in COMMON_ENGLISH_WORDS
    assert "monday" in COMMON_ENGLISH_WORDS


def test_common_english_words_is_lowercase():
    for word in COMMON_ENGLISH_WORDS:
        assert word == word.lower(), f"{word} should be lowercase"


# ── EntityRegistry creation and empty state ─────────────────────────────


def test_load_from_nonexistent_dir(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    assert registry.people == {}
    assert registry.projects == []
    assert registry.mode == "personal"
    assert registry.ambiguous_flags == []


def test_save_and_load_roundtrip(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="work",
        people=[{"name": "Alice", "relationship": "colleague", "context": "work"}],
        projects=["MemPalace"],
    )
    # Load again from same dir
    loaded = EntityRegistry.load(config_dir=tmp_path)
    assert loaded.mode == "work"
    assert "Alice" in loaded.people
    assert "MemPalace" in loaded.projects


def test_save_creates_file(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.save()
    assert (tmp_path / "entity_registry.json").exists()


def test_load_recovers_from_corrupt_json(tmp_path):
    """A malformed registry file falls back to the empty default schema."""
    (tmp_path / "entity_registry.json").write_text("{not json")
    registry = EntityRegistry.load(config_dir=tmp_path)
    assert registry.people == {}
    assert registry.projects == []


# ── seed ────────────────────────────────────────────────────────────────


def test_seed_registers_people(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
            {"name": "Devon", "relationship": "friend", "context": "personal"},
        ],
        projects=["MemPalace"],
    )
    assert "Riley" in registry.people
    assert "Devon" in registry.people
    assert registry.people["Riley"]["relationship"] == "daughter"
    assert registry.people["Riley"]["source"] == "onboarding"
    assert registry.people["Riley"]["confidence"] == 1.0


def test_seed_registers_projects(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="work", people=[], projects=["Acme", "Widget"])
    assert registry.projects == ["Acme", "Widget"]


def test_seed_sets_mode(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="combo", people=[], projects=[])
    assert registry.mode == "combo"


def test_seed_flags_ambiguous_names(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Grace", "relationship": "friend", "context": "personal"},
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
        ],
        projects=[],
    )
    assert "grace" in registry.ambiguous_flags
    # Riley is not a common English word
    assert "riley" not in registry.ambiguous_flags


def test_seed_with_aliases(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Maxwell", "relationship": "friend", "context": "personal"}],
        projects=[],
        aliases={"Max": "Maxwell"},
    )
    assert "Maxwell" in registry.people
    assert "Max" in registry.people
    assert registry.people["Max"].get("canonical") == "Maxwell"


def test_seed_skips_empty_names(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "", "relationship": "", "context": "personal"}],
        projects=[],
    )
    assert len(registry.people) == 0


# ── iter_known_names (spellcheck integration) ──────────────────────────


def test_iter_known_names_includes_canonical_and_aliases(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
            {"name": "Maxwell", "relationship": "friend", "context": "personal"},
        ],
        projects=[],
        aliases={"Max": "Maxwell"},
    )
    names = registry.iter_known_names()
    # All entries are lowercased so the spellchecker can do case-insensitive
    # word boundary matching against user text without re-normalizing.
    assert "riley" in names
    assert "maxwell" in names
    assert "max" in names
    assert all(n == n.lower() for n in names)


def test_iter_known_names_empty_registry(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    assert registry.iter_known_names() == set()


# ── summary ─────────────────────────────────────────────────────────────


def test_summary(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=["MemPalace"],
    )
    s = registry.summary()
    assert "personal" in s
    assert "Riley" in s
    assert "MemPalace" in s
