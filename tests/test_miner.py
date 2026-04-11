from pathlib import Path

import chromadb
import yaml

from mempalace.miner import mine, scan_project


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scanned_files(project_root: Path, **kwargs):
    files = scan_project(str(project_root), **kwargs)
    return sorted(path.relative_to(project_root).as_posix() for path in files)


def test_project_mining(tmp_path):
    project_root = tmp_path.resolve()
    (project_root / "backend").mkdir(parents=True)

    write_file(project_root / "backend" / "app.py", "def main():\n    print('hello world')\n" * 20)
    (project_root / "mempalace.yaml").write_text(
        yaml.dump(
            {
                "wing": "test_project",
                "rooms": [
                    {"name": "backend", "description": "Backend code"},
                    {"name": "general", "description": "General"},
                ],
            }
        )
    )

    palace_path = project_root / "palace"
    mine(str(project_root), str(palace_path))

    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_collection("mempalace_drawers")
    assert col.count() > 0

    # The miner must also populate the colocated trie index so keyword
    # / temporal search works immediately after mining.
    from mempalace.trie_index import TrieIndex, trie_db_path

    trie_db = trie_db_path(str(palace_path))
    assert Path(trie_db).is_dir(), "trie_index.lmdb should exist after mining"
    stats = TrieIndex(db_path=trie_db).stats()
    assert stats["postings"] > 0
    assert stats["unique_drawers"] == col.count()


def test_scan_project_respects_gitignore(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "ignored.py\ngenerated/\n")
    write_file(project_root / "src" / "app.py", "print('hello')\n" * 20)
    write_file(project_root / "ignored.py", "print('ignore me')\n" * 20)
    write_file(project_root / "generated" / "artifact.py", "print('artifact')\n" * 20)

    assert scanned_files(project_root) == ["src/app.py"]


def test_scan_project_respects_nested_gitignore(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "*.log\n")
    write_file(project_root / "subrepo" / ".gitignore", "tasks/\n")
    write_file(project_root / "subrepo" / "src" / "main.py", "print('main')\n" * 20)
    write_file(project_root / "subrepo" / "tasks" / "task.py", "print('task')\n" * 20)
    write_file(project_root / "subrepo" / "debug.log", "debug\n" * 20)

    assert scanned_files(project_root) == ["subrepo/src/main.py"]


def test_scan_project_allows_nested_gitignore_override(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "*.csv\n")
    write_file(project_root / "subrepo" / ".gitignore", "!keep.csv\n")
    write_file(project_root / "drop.csv", "a,b,c\n" * 20)
    write_file(project_root / "subrepo" / "keep.csv", "a,b,c\n" * 20)

    assert scanned_files(project_root) == ["subrepo/keep.csv"]


def test_scan_project_allows_gitignore_negation_when_parent_dir_is_visible(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "generated/*\n!generated/keep.py\n")
    write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
    write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

    assert scanned_files(project_root) == ["generated/keep.py"]


def test_scan_project_does_not_reinclude_file_from_ignored_directory(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "generated/\n!generated/keep.py\n")
    write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
    write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

    assert scanned_files(project_root) == []


def test_scan_project_can_disable_gitignore(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "data/\n")
    write_file(project_root / "data" / "stuff.csv", "a,b,c\n" * 20)

    assert scanned_files(project_root, respect_gitignore=False) == ["data/stuff.csv"]


def test_scan_project_can_include_ignored_directory(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "docs/\n")
    write_file(project_root / "docs" / "guide.md", "# Guide\n" * 20)

    assert scanned_files(project_root, include_ignored=["docs"]) == ["docs/guide.md"]


def test_scan_project_can_include_specific_ignored_file(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "generated/\n")
    write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
    write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

    assert scanned_files(project_root, include_ignored=["generated/keep.py"]) == [
        "generated/keep.py"
    ]


def test_scan_project_can_include_exact_file_without_known_extension(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".gitignore", "README\n")
    write_file(project_root / "README", "hello\n" * 20)

    assert scanned_files(project_root, include_ignored=["README"]) == ["README"]


def test_scan_project_include_override_beats_skip_dirs(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)

    assert scanned_files(
        project_root,
        respect_gitignore=False,
        include_ignored=[".pytest_cache"],
    ) == [".pytest_cache/cache.py"]


def test_scan_project_skip_dirs_still_apply_without_override(tmp_path):
    project_root = tmp_path.resolve()

    write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)
    write_file(project_root / "main.py", "print('main')\n" * 20)

    assert scanned_files(project_root, respect_gitignore=False) == ["main.py"]


def test_scan_project_skips_symlinks(tmp_path):
    """Symlinks are skipped so scan_project can't follow them to /dev/urandom."""
    project_root = tmp_path.resolve()
    write_file(project_root / "real.py", "print('real')\n" * 20)
    (project_root / "link.py").symlink_to(project_root / "real.py")
    files = scanned_files(project_root, respect_gitignore=False)
    # real.py is present, the symlink is not (symlinks are skipped).
    assert files == ["real.py"]


def test_scan_project_respects_max_file_size(tmp_path, monkeypatch):
    """Files over MAX_FILE_SIZE are skipped without opening them."""
    import mempalace.miner as miner_mod

    monkeypatch.setattr(miner_mod, "MAX_FILE_SIZE", 64)  # tiny for testing
    project_root = tmp_path.resolve()
    write_file(project_root / "tiny.py", "a = 1\n")
    write_file(project_root / "big.py", "b = 1\n" * 100)  # 600+ bytes
    files = scanned_files(project_root, respect_gitignore=False)
    assert "tiny.py" in files
    assert "big.py" not in files


def test_status_on_missing_palace(tmp_path, capsys):
    """status() prints the 'no palace' message when the collection can't open."""
    from mempalace.miner import status

    status(palace_path=str(tmp_path / "nonexistent"))
    out = capsys.readouterr().out
    assert "No palace found" in out


def test_status_on_seeded_palace(tmp_path, capsys):
    """status() tallies drawers by wing and room from a live collection."""
    from mempalace.miner import status

    palace = tmp_path / "palace"
    palace.mkdir()
    client = chromadb.PersistentClient(path=str(palace))
    col = client.get_or_create_collection("mempalace_drawers")
    col.add(
        ids=["d1", "d2", "d3"],
        documents=["alpha", "beta", "gamma"],
        metadatas=[
            {"wing": "project", "room": "backend"},
            {"wing": "project", "room": "backend"},
            {"wing": "notes", "room": "planning"},
        ],
    )
    del col, client

    status(palace_path=str(palace))
    out = capsys.readouterr().out
    assert "3 drawers" in out
    assert "project" in out
    assert "backend" in out
    assert "planning" in out


def test_file_already_mined_unmined_returns_false(tmp_path):
    """A file that was never filed returns False."""
    from mempalace.miner import file_already_mined

    palace = tmp_path / "palace"
    palace.mkdir()
    client = chromadb.PersistentClient(path=str(palace))
    col = client.get_or_create_collection("mempalace_drawers")

    src = tmp_path / "src.py"
    src.write_text("hello")
    assert file_already_mined(col, str(src)) is False


def test_file_already_mined_matching_mtime_returns_true(tmp_path):
    """A file that's been filed with the same mtime returns True."""
    from mempalace.miner import file_already_mined

    palace = tmp_path / "palace"
    palace.mkdir()
    client = chromadb.PersistentClient(path=str(palace))
    col = client.get_or_create_collection("mempalace_drawers")

    src = tmp_path / "src.py"
    src.write_text("hello")
    mtime = src.stat().st_mtime

    col.add(
        ids=["d1"],
        documents=["hello"],
        metadatas=[{"source_file": str(src), "source_mtime": str(mtime)}],
    )
    assert file_already_mined(col, str(src)) is True


def test_file_already_mined_stale_mtime_returns_false(tmp_path):
    """A file whose mtime no longer matches the stored mtime returns False."""
    from mempalace.miner import file_already_mined

    palace = tmp_path / "palace"
    palace.mkdir()
    client = chromadb.PersistentClient(path=str(palace))
    col = client.get_or_create_collection("mempalace_drawers")

    src = tmp_path / "src.py"
    src.write_text("hello")

    col.add(
        ids=["d1"],
        documents=["hello"],
        metadatas=[{"source_file": str(src), "source_mtime": "0.0"}],  # stale
    )
    assert file_already_mined(col, str(src)) is False
