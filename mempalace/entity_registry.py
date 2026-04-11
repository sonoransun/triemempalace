#!/usr/bin/env python3
"""
entity_registry.py — Persistent personal entity registry for MemPalace.

Stores user-confirmed names, projects, aliases, and a known-names set
used by the spellchecker. Populated from onboarding via :meth:`seed`
and read by :mod:`mempalace.spellcheck` via :meth:`iter_known_names`.

Usage:
    from mempalace.entity_registry import EntityRegistry
    registry = EntityRegistry.load()
    registry.seed(mode="personal", people=[...], projects=[...])
    print(registry.summary())
"""

import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Common English words that could be confused with names
# These get flagged as AMBIGUOUS and require context disambiguation
# ─────────────────────────────────────────────────────────────────────────────

COMMON_ENGLISH_WORDS = {
    # Words that are also common personal names
    "ever",
    "grace",
    "will",
    "bill",
    "mark",
    "april",
    "may",
    "june",
    "joy",
    "hope",
    "faith",
    "chance",
    "chase",
    "hunter",
    "dash",
    "flash",
    "star",
    "sky",
    "river",
    "brook",
    "lane",
    "art",
    "clay",
    "gil",
    "nat",
    "max",
    "rex",
    "ray",
    "jay",
    "rose",
    "violet",
    "lily",
    "ivy",
    "ash",
    "reed",
    "sage",
    # Words that look like names at start of sentence
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

# ─────────────────────────────────────────────────────────────────────────────
# Entity Registry
# ─────────────────────────────────────────────────────────────────────────────


class EntityRegistry:
    """
    Persistent personal entity registry.

    Stored at ~/.mempalace/entity_registry.json
    Schema:
    {
      "mode": "personal",   # work | personal | combo
      "version": 1,
      "people": {
        "Riley": {
          "source": "onboarding",
          "contexts": ["personal"],
          "aliases": [],
          "relationship": "daughter",
          "confidence": 1.0
        }
      },
      "projects": ["MemPalace", "Acme"],
      "ambiguous_flags": ["riley", "max"]
    }
    """

    DEFAULT_PATH = Path.home() / ".mempalace" / "entity_registry.json"

    def __init__(self, data: dict, path: Path):
        self._data = data
        self._path = path

    # ── Load / Save ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "EntityRegistry":
        path = (Path(config_dir) / "entity_registry.json") if config_dir else cls.DEFAULT_PATH
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(data, path)
            except (json.JSONDecodeError, OSError):
                pass
        return cls(cls._empty(), path)

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    @staticmethod
    def _empty() -> dict:
        return {
            "version": 1,
            "mode": "personal",
            "people": {},
            "projects": [],
            "ambiguous_flags": [],
        }

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._data.get("mode", "personal")

    @property
    def people(self) -> dict:
        return self._data.get("people", {})

    @property
    def projects(self) -> list:
        return self._data.get("projects", [])

    @property
    def ambiguous_flags(self) -> list:
        return self._data.get("ambiguous_flags", [])

    # ── Seed from onboarding ─────────────────────────────────────────────────

    def seed(self, mode: str, people: list, projects: list, aliases: dict = None):
        """
        Seed the registry from onboarding data.

        people: list of dicts {"name": str, "relationship": str, "context": str}
        projects: list of str
        aliases: dict {"Max": "Maxwell", ...}
        """
        self._data["mode"] = mode
        self._data["projects"] = list(projects)

        aliases = aliases or {}
        reverse_aliases = {v: k for k, v in aliases.items()}  # Maxwell → Max

        for entry in people:
            name = entry["name"].strip()
            if not name:
                continue
            context = entry.get("context", "personal")
            relationship = entry.get("relationship", "")

            self._data["people"][name] = {
                "source": "onboarding",
                "contexts": [context],
                "aliases": [reverse_aliases[name]] if name in reverse_aliases else [],
                "relationship": relationship,
                "confidence": 1.0,
            }

            # Also register aliases
            if name in reverse_aliases:
                alias = reverse_aliases[name]
                self._data["people"][alias] = {
                    "source": "onboarding",
                    "contexts": [context],
                    "aliases": [name],
                    "relationship": relationship,
                    "confidence": 1.0,
                    "canonical": name,
                }

        # Flag ambiguous names (also common English words)
        ambiguous = []
        for name in self._data["people"]:
            if name.lower() in COMMON_ENGLISH_WORDS:
                ambiguous.append(name.lower())
        self._data["ambiguous_flags"] = ambiguous

        self.save()

    # ── Known-names accessor for spellcheck ──────────────────────────────────

    def iter_known_names(self) -> set[str]:
        """Return lowercased canonical names + aliases for spellcheck protection.

        Used by :func:`mempalace.spellcheck._load_known_names` to build the
        set of words the autocorrector must NOT touch (proper nouns, project
        codenames, nicknames). Replaces the previous private-attribute reach
        into ``reg._data``.
        """
        names: set[str] = set()
        for canonical, info in self.people.items():
            names.add(canonical.lower())
            for alias in info.get("aliases", []):
                names.add(alias.lower())
        return names

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Mode: {self.mode}",
            f"People: {len(self.people)} ({', '.join(list(self.people.keys())[:8])}{'...' if len(self.people) > 8 else ''})",
            f"Projects: {', '.join(self.projects) or '(none)'}",
            f"Ambiguous flags: {', '.join(self.ambiguous_flags) or '(none)'}",
        ]
        return "\n".join(lines)
