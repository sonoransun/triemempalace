"""Tests for mempalace.cli — the main CLI dispatcher."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mempalace.cli import (
    cmd_compress,
    cmd_hook,
    cmd_init,
    cmd_instructions,
    cmd_kg_extract,
    cmd_mine,
    cmd_models,
    cmd_repair,
    cmd_rerankers,
    cmd_search,
    cmd_split,
    cmd_status,
    cmd_trie_repair,
    cmd_wakeup,
    main,
)

# ── cmd_status ─────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_status_default_palace(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None)
    mock_miner = MagicMock()
    with patch.dict("sys.modules", {"mempalace.miner": mock_miner}):
        cmd_status(args)
        mock_miner.status.assert_called_once_with(palace_path="/fake/palace")


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_status_custom_palace(mock_config_cls):
    args = argparse.Namespace(palace="~/my_palace")
    mock_miner = MagicMock()
    with patch.dict("sys.modules", {"mempalace.miner": mock_miner}):
        cmd_status(args)
        import os

        expected = os.path.expanduser("~/my_palace")
        mock_miner.status.assert_called_once_with(palace_path=expected)


# ── cmd_search ─────────────────────────────────────────────────────────


def _search_args(**overrides):
    """Build a search-subcommand Namespace with all trie/temporal/model/compress flags defaulted."""
    defaults = dict(
        palace=None,
        query="test query",
        wing=None,
        room=None,
        results=5,
        keyword=[],
        keyword_prefix=[],
        since=None,
        until=None,
        as_of=None,
        warm_trie=False,
        model=None,
        compress="auto",
        token_budget=None,
        dup_threshold=0.7,
        sent_threshold=0.75,
        novelty_threshold=0.2,
        rerank="none",
        rerank_prune=True,
        enable_kg_ppr=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_search_calls_search(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = _search_args(query="test query", wing="mywing", room="myroom", results=3)
    with patch("mempalace.searcher.search") as mock_search:
        cmd_search(args)
        mock_search.assert_called_once_with(
            query="test query",
            palace_path="/fake/palace",
            wing="mywing",
            room="myroom",
            n_results=3,
            keywords=None,
            keyword_mode="all",
            since=None,
            until=None,
            as_of=None,
            model=None,
            compress="auto",
            token_budget=None,
            dup_threshold=0.7,
            sent_threshold=0.75,
            novelty_threshold=0.2,
            rerank=None,
            rerank_prune=True,
            enable_kg_ppr=False,
        )


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_search_error_exits(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = _search_args(query="q")
    from mempalace.searcher import SearchError

    with patch("mempalace.searcher.search", side_effect=SearchError("fail")):
        with pytest.raises(SystemExit) as exc_info:
            cmd_search(args)
        assert exc_info.value.code == 1


# ── cmd_instructions ───────────────────────────────────────────────────


def test_cmd_instructions_calls_run_instructions():
    args = argparse.Namespace(name="help")
    with patch("mempalace.instructions_cli.run_instructions") as mock_run:
        cmd_instructions(args)
        mock_run.assert_called_once_with(name="help")


# ── cmd_hook ───────────────────────────────────────────────────────────


def test_cmd_hook_calls_run_hook():
    args = argparse.Namespace(hook="session-start", harness="claude-code")
    with patch("mempalace.hooks_cli.run_hook") as mock_run:
        cmd_hook(args)
        mock_run.assert_called_once_with(hook_name="session-start", harness="claude-code")


# ── cmd_init ───────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_init_no_entities(mock_config_cls, tmp_path):
    args = argparse.Namespace(dir=str(tmp_path), yes=True)
    with (
        patch("mempalace.entity_detector.scan_for_detection", return_value=[]),
        patch("mempalace.room_detector_local.detect_rooms_local") as mock_rooms,
    ):
        cmd_init(args)
        mock_rooms.assert_called_once_with(project_dir=str(tmp_path), yes=True)
        mock_config_cls.return_value.init.assert_called_once()


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_init_with_entities(mock_config_cls, tmp_path):
    fake_files = [tmp_path / "a.txt"]
    detected = {"people": [{"name": "Alice"}], "projects": [], "uncertain": []}
    confirmed = {"people": ["Alice"], "projects": []}
    args = argparse.Namespace(dir=str(tmp_path), yes=True)
    with (
        patch("mempalace.entity_detector.scan_for_detection", return_value=fake_files),
        patch("mempalace.entity_detector.detect_entities", return_value=detected),
        patch("mempalace.entity_detector.confirm_entities", return_value=confirmed),
        patch("mempalace.room_detector_local.detect_rooms_local"),
        patch("builtins.open", MagicMock()),
    ):
        cmd_init(args)


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_init_with_entities_zero_total(mock_config_cls, tmp_path, capsys):
    """When entities detected but total is 0, prints 'No entities' message."""
    fake_files = [tmp_path / "a.txt"]
    detected = {"people": [], "projects": [], "uncertain": []}
    args = argparse.Namespace(dir=str(tmp_path), yes=False)
    with (
        patch("mempalace.entity_detector.scan_for_detection", return_value=fake_files),
        patch("mempalace.entity_detector.detect_entities", return_value=detected),
        patch("mempalace.room_detector_local.detect_rooms_local"),
    ):
        cmd_init(args)
    out = capsys.readouterr().out
    assert "No entities detected" in out


# ── cmd_mine ───────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_projects_mode(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/src",
        palace=None,
        mode="projects",
        wing=None,
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=[],
        extract="exchange",
        model=None,
    )
    with patch("mempalace.miner.mine") as mock_mine:
        cmd_mine(args)
        mock_mine.assert_called_once_with(
            project_dir="/src",
            palace_path="/fake/palace",
            wing_override=None,
            agent="mempalace",
            limit=0,
            dry_run=False,
            respect_gitignore=True,
            include_ignored=[],
            model=None,
        )


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_convos_mode(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/chats",
        palace=None,
        mode="convos",
        wing="mywing",
        agent="me",
        limit=10,
        dry_run=True,
        no_gitignore=False,
        include_ignored=[],
        extract="general",
        model=None,
    )
    with patch("mempalace.convo_miner.mine_convos") as mock_mine:
        cmd_mine(args)
        mock_mine.assert_called_once_with(
            convo_dir="/chats",
            palace_path="/fake/palace",
            wing="mywing",
            agent="me",
            limit=10,
            dry_run=True,
            extract_mode="general",
            model=None,
        )


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_include_ignored_comma_split(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/src",
        palace=None,
        mode="projects",
        wing=None,
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=["a.txt,b.txt", "c.txt"],
        extract="exchange",
    )
    with patch("mempalace.miner.mine") as mock_mine:
        cmd_mine(args)
        mock_mine.assert_called_once()
        call_kwargs = mock_mine.call_args[1]
        assert call_kwargs["include_ignored"] == ["a.txt", "b.txt", "c.txt"]


# ── cmd_wakeup ─────────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_wakeup(mock_config_cls, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None, wing=None)
    mock_stack = MagicMock()
    mock_stack.wake_up.return_value = "Hello world context"
    with patch("mempalace.layers.MemoryStack", return_value=mock_stack):
        cmd_wakeup(args)
    out = capsys.readouterr().out
    assert "Hello world context" in out
    assert "tokens" in out


# ── cmd_split ──────────────────────────────────────────────────────────


def test_cmd_split_basic():
    args = argparse.Namespace(dir="/chats", output_dir=None, dry_run=False, min_sessions=2)
    with patch("mempalace.split_mega_files.main") as mock_main:
        cmd_split(args)
        mock_main.assert_called_once()


def test_cmd_split_all_options():
    args = argparse.Namespace(dir="/chats", output_dir="/out", dry_run=True, min_sessions=5)
    with patch("mempalace.split_mega_files.main") as mock_main:
        cmd_split(args)
        mock_main.assert_called_once()
    # sys.argv should be restored
    assert sys.argv[0] != "mempalace split"


# ── main() argparse dispatch ──────────────────────────────────────────


def test_main_no_args_prints_help(capsys):
    with patch("sys.argv", ["mempalace"]):
        main()
    out = capsys.readouterr().out
    assert "MemPalace" in out


def test_main_status_dispatches():
    with (
        patch("sys.argv", ["mempalace", "status"]),
        patch("mempalace.cli.cmd_status") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_search_dispatches():
    with (
        patch("sys.argv", ["mempalace", "search", "my query"]),
        patch("mempalace.cli.cmd_search") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_init_dispatches():
    with (
        patch("sys.argv", ["mempalace", "init", "/some/dir"]),
        patch("mempalace.cli.cmd_init") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_mine_dispatches():
    with (
        patch("sys.argv", ["mempalace", "mine", "/some/dir"]),
        patch("mempalace.cli.cmd_mine") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_wakeup_dispatches():
    with (
        patch("sys.argv", ["mempalace", "wake-up"]),
        patch("mempalace.cli.cmd_wakeup") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_split_dispatches():
    with (
        patch("sys.argv", ["mempalace", "split", "/chats"]),
        patch("mempalace.cli.cmd_split") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_mcp_command_prints_setup_guidance(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mempalace", "mcp"])

    main()

    captured = capsys.readouterr()
    assert "MemPalace MCP quick setup:" in captured.out
    assert "claude mcp add mempalace -- python -m mempalace.mcp_server" in captured.out
    assert "\nOptional custom palace:\n" in captured.out
    assert "python -m mempalace.mcp_server --palace /path/to/palace" in captured.out
    assert "[--palace /path/to/palace]" not in captured.out
    assert captured.err == ""


def test_mcp_command_uses_custom_palace_path_when_provided(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mempalace", "--palace", "~/tmp/my palace", "mcp"])

    main()

    captured = capsys.readouterr()
    expanded = str(Path("~/tmp/my palace").expanduser())

    assert "python -m mempalace.mcp_server --palace" in captured.out
    assert expanded in captured.out
    assert "Optional custom palace:" not in captured.out
    assert "[--palace /path/to/palace]" not in captured.out
    assert captured.err == ""


def test_main_hook_no_subcommand_prints_help(capsys):
    with patch("sys.argv", ["mempalace", "hook"]):
        main()
    out = capsys.readouterr().out
    assert "hook" in out.lower() or "run" in out.lower()


def test_main_hook_run_dispatches():
    with (
        patch(
            "sys.argv",
            ["mempalace", "hook", "run", "--hook", "session-start", "--harness", "claude-code"],
        ),
        patch("mempalace.cli.cmd_hook") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_instructions_no_subcommand_prints_help(capsys):
    with patch("sys.argv", ["mempalace", "instructions"]):
        main()
    out = capsys.readouterr().out
    assert "instructions" in out.lower() or "init" in out.lower()


def test_main_instructions_dispatches():
    with (
        patch("sys.argv", ["mempalace", "instructions", "help"]),
        patch("mempalace.cli.cmd_instructions") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_repair_dispatches():
    with (
        patch("sys.argv", ["mempalace", "repair"]),
        patch("mempalace.cli.cmd_repair") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


def test_main_compress_dispatches():
    with (
        patch("sys.argv", ["mempalace", "compress"]),
        patch("mempalace.cli.cmd_compress") as mock_cmd,
    ):
        main()
        mock_cmd.assert_called_once()


# ── cmd_repair ─────────────────────────────────────────────────────────


def _config_with_default_models(palace_path: str) -> MagicMock:
    """Build a MempalaceConfig stub that exposes the multi-model fields cmd_repair needs."""
    cfg = MagicMock()
    cfg.palace_path = palace_path
    cfg.enabled_embedding_models = ["default"]
    cfg.default_embedding_model = "default"
    return cfg


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_no_palace(mock_config_cls, tmp_path, capsys):
    mock_config_cls.return_value = _config_with_default_models(str(tmp_path / "nonexistent"))
    cmd_repair(argparse.Namespace(palace=None))
    out = capsys.readouterr().out
    assert "No palace found" in out


def _seed_palace_db(palace_dir):
    """cmd_repair gates on a real chroma.sqlite3 sentinel — the tests
    don't need a valid DB, just the file's presence."""
    (palace_dir / "chroma.sqlite3").write_text("")


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_error_reading(mock_config_cls, mock_open, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    _seed_palace_db(palace_dir)
    mock_config_cls.return_value = _config_with_default_models(str(palace_dir))
    mock_open.side_effect = ValueError("corrupt db")
    cmd_repair(argparse.Namespace(palace=None))
    out = capsys.readouterr().out
    assert "[default] skipped" in out
    assert "Nothing to repair" in out


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_zero_drawers(mock_config_cls, mock_open, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    _seed_palace_db(palace_dir)
    mock_config_cls.return_value = _config_with_default_models(str(palace_dir))
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_open.return_value = mock_col
    cmd_repair(argparse.Namespace(palace=None))
    out = capsys.readouterr().out
    assert "Nothing to repair" in out


@patch("mempalace.palace_io.drop_collection_cache")
@patch("mempalace.palace_io.delete_collection")
@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_repair_success(
    mock_config_cls, mock_open, mock_delete, mock_drop_cache, tmp_path, capsys
):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    _seed_palace_db(palace_dir)
    mock_config_cls.return_value = _config_with_default_models(str(palace_dir))

    old_col = MagicMock()
    old_col.count.return_value = 2
    old_col.get.return_value = {
        "ids": ["id1", "id2"],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"wing": "a"}, {"wing": "b"}],
    }
    new_col = MagicMock()
    mock_open.side_effect = [old_col, new_col]

    cmd_repair(argparse.Namespace(palace=None))
    out = capsys.readouterr().out

    assert "Repair complete" in out
    assert "2 drawers rebuilt" in out
    mock_delete.assert_called_once()
    mock_drop_cache.assert_called_once()
    new_col.add.assert_called_once()


# ── cmd_compress ───────────────────────────────────────────────────────


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_compress_no_palace(mock_config_cls, mock_open, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None, wing=None, dry_run=False, config=None)
    mock_open.side_effect = ValueError("no palace")
    with pytest.raises(SystemExit):
        cmd_compress(args)


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_compress_no_drawers(mock_config_cls, mock_open, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None, wing="mywing", dry_run=False, config=None)
    mock_col = MagicMock()
    mock_col.get.return_value = {"documents": [], "metadatas": [], "ids": []}
    mock_open.return_value = mock_col
    cmd_compress(args)
    out = capsys.readouterr().out
    assert "No drawers found" in out


def _make_mock_dialect_module(dialect_instance):
    """Create a mock dialect module with a Dialect class that returns the given instance."""
    mock_mod = MagicMock()
    mock_mod.Dialect.return_value = dialect_instance
    mock_mod.Dialect.from_config.return_value = dialect_instance
    mock_mod.Dialect.count_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
    return mock_mod


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_compress_dry_run(mock_config_cls, mock_open, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None, wing=None, dry_run=True, config=None)
    mock_col = MagicMock()
    mock_col.get.side_effect = [
        {
            "documents": ["some long text here for testing"],
            "metadatas": [{"wing": "test", "room": "general", "source_file": "test.txt"}],
            "ids": ["id1"],
        },
        {"documents": [], "metadatas": [], "ids": []},
    ]
    mock_open.return_value = mock_col

    mock_dialect = MagicMock()
    mock_dialect.compress.return_value = "compressed"
    mock_dialect.compression_stats.return_value = {
        "original_chars": 100,
        "summary_chars": 30,
        "original_tokens_est": 25,
        "summary_tokens_est": 8,
        "size_ratio": 3.3,
        "note": "Estimates only.",
    }
    mock_dialect_mod = _make_mock_dialect_module(mock_dialect)

    with patch.dict("sys.modules", {"mempalace.dialect": mock_dialect_mod}):
        cmd_compress(args)
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    assert "Compressing" in out
    assert "Total:" in out


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_compress_with_config(mock_config_cls, mock_open, tmp_path, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    config_file = tmp_path / "entities.json"
    config_file.write_text('{"people": [], "projects": []}')
    args = argparse.Namespace(palace=None, wing=None, dry_run=True, config=str(config_file))
    mock_col = MagicMock()
    mock_col.get.return_value = {"documents": [], "metadatas": [], "ids": []}
    mock_open.return_value = mock_col

    mock_dialect = MagicMock()
    mock_dialect_mod = _make_mock_dialect_module(mock_dialect)

    with patch.dict("sys.modules", {"mempalace.dialect": mock_dialect_mod}):
        cmd_compress(args)
    out = capsys.readouterr().out
    assert "Loaded entity config" in out


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_compress_stores_results(mock_config_cls, mock_open, capsys):
    """Non-dry-run compress stores to mempalace_compressed collection."""
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(palace=None, wing=None, dry_run=False, config=None)
    read_col = MagicMock()
    read_col.get.side_effect = [
        {
            "documents": ["text"],
            "metadatas": [{"wing": "w", "room": "r", "source_file": "f.txt"}],
            "ids": ["id1"],
        },
        {"documents": [], "metadatas": [], "ids": []},
    ]
    comp_col = MagicMock()
    # cmd_compress calls open_collection twice: once for the read side
    # (default model) and once for the compressed sidecar collection.
    mock_open.side_effect = [read_col, comp_col]

    mock_dialect = MagicMock()
    mock_dialect.compress.return_value = "compressed"
    mock_dialect.compression_stats.return_value = {
        "original_chars": 100,
        "summary_chars": 30,
        "original_tokens_est": 25,
        "summary_tokens_est": 8,
        "size_ratio": 3.3,
        "note": "Estimates only.",
    }
    mock_dialect_mod = _make_mock_dialect_module(mock_dialect)

    with patch.dict("sys.modules", {"mempalace.dialect": mock_dialect_mod}):
        cmd_compress(args)
    out = capsys.readouterr().out
    assert "Stored" in out
    comp_col.upsert.assert_called_once()
    # The second open_collection call must use the compressed override.
    second_call = mock_open.call_args_list[1]
    assert second_call.kwargs.get("collection_name_override") == "mempalace_compressed"


def test_cmd_repair_trailing_slash_does_not_recurse(tmp_path):
    """Repair with trailing slash should put backup outside palace dir (#395)."""
    import os

    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    palace_arg = str(palace_dir) + "/"
    args = argparse.Namespace(palace=palace_arg)
    with patch("mempalace.cli.MempalaceConfig") as mock_config_cls:
        mock_config_cls.return_value = _config_with_default_models(str(palace_dir))
        with patch("mempalace.palace_io.open_collection") as mock_open:
            empty = MagicMock()
            empty.count.return_value = 0
            mock_open.return_value = empty
            cmd_repair(args)
    palace_path = os.path.expanduser(palace_arg).rstrip(os.sep)
    backup_path = palace_path + ".backup"
    assert not backup_path.startswith(palace_path + os.sep)


# ── cmd_trie_repair ───────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_trie_repair_no_palace(mock_config_cls, tmp_path, capsys):
    mock_config_cls.return_value.palace_path = str(tmp_path / "nonexistent")
    cmd_trie_repair(argparse.Namespace(palace=None))
    out = capsys.readouterr().out
    assert "No palace found" in out


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_trie_repair_open_failure(mock_config_cls, mock_open, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    mock_config_cls.return_value.palace_path = str(palace_dir)
    mock_open.side_effect = ValueError("bad palace")
    cmd_trie_repair(argparse.Namespace(palace=None))
    out = capsys.readouterr().out
    assert "Could not open palace" in out


@patch("mempalace.palace_io.open_collection")
@patch("mempalace.cli.MempalaceConfig")
def test_cmd_trie_repair_success(mock_config_cls, mock_open, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    mock_config_cls.return_value.palace_path = str(palace_dir)

    mock_col = MagicMock()
    mock_col.count.return_value = 5
    mock_open.return_value = mock_col

    fake_trie = MagicMock()
    fake_trie.rebuild_from_collection.return_value = 17
    fake_trie.stats.return_value = {
        "unique_tokens": 4,
        "unique_drawers": 5,
        "db_path": str(palace_dir / "trie_index.lmdb"),
    }
    with patch("mempalace.trie_index.TrieIndex", return_value=fake_trie):
        cmd_trie_repair(argparse.Namespace(palace=None))
    out = capsys.readouterr().out
    assert "Postings filed" in out
    assert "17" in out


# ── cmd_models ────────────────────────────────────────────────────────


def _models_config(default="default", enabled=("default",)) -> MagicMock:
    cfg = MagicMock()
    cfg.palace_path = "/fake/palace"
    cfg.default_embedding_model = default
    cfg.enabled_embedding_models = list(enabled)
    return cfg


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_list(mock_config_cls, capsys):
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="list")
    with patch("mempalace.palace_io.open_collection") as mock_open:
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_open.return_value = mock_col
        cmd_models(args)
    out = capsys.readouterr().out
    assert "SLUG" in out
    assert "default model:" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_list_open_failure_shows_zero(mock_config_cls, capsys):
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="list")
    with patch("mempalace.palace_io.open_collection", side_effect=RuntimeError("boom")):
        cmd_models(args)
    out = capsys.readouterr().out
    assert "default model:" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_list_default_action(mock_config_cls, capsys):
    """Passing models_action=None defaults to 'list' behavior."""
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action=None)
    with patch("mempalace.palace_io.open_collection") as mock_open:
        mock_open.return_value = MagicMock(count=lambda: 0)
        cmd_models(args)
    out = capsys.readouterr().out
    assert "SLUG" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_missing_slug_exits(mock_config_cls):
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="enable", slug=None)
    with pytest.raises(SystemExit) as exc_info:
        cmd_models(args)
    assert exc_info.value.code == 2


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_unknown_slug_exits(mock_config_cls, capsys):
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="enable", slug="not-a-real-slug")
    with patch(
        "mempalace.embeddings.get_spec", side_effect=KeyError("unknown slug not-a-real-slug")
    ), pytest.raises(SystemExit) as exc_info:
        cmd_models(args)
    assert exc_info.value.code == 2


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_enable(mock_config_cls, capsys):
    cfg = _models_config()
    mock_config_cls.return_value = cfg
    args = argparse.Namespace(models_action="enable", slug="fake")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(slug="fake", backend="fake", extras_required=())
        cmd_models(args)
    cfg.save_embedding_config.assert_called_once()


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_disable_rejects_default(mock_config_cls, capsys):
    cfg = _models_config()
    mock_config_cls.return_value = cfg
    args = argparse.Namespace(models_action="disable", slug="default")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(slug="default", backend="chroma-default", extras_required=())
        with pytest.raises(SystemExit) as exc_info:
            cmd_models(args)
    assert exc_info.value.code == 2


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_disable_other(mock_config_cls, capsys):
    cfg = _models_config(enabled=("default", "fake"))
    mock_config_cls.return_value = cfg
    args = argparse.Namespace(models_action="disable", slug="fake")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(slug="fake", backend="fake", extras_required=())
        cmd_models(args)
    cfg.save_embedding_config.assert_called_once()


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_set_default(mock_config_cls, capsys):
    cfg = _models_config()
    mock_config_cls.return_value = cfg
    args = argparse.Namespace(models_action="set-default", slug="fake")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(slug="fake", backend="fake", extras_required=())
        cmd_models(args)
    cfg.save_embedding_config.assert_called_once()


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_download_missing_extras(mock_config_cls, capsys):
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="download", slug="fake")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(
            slug="fake", backend="fastembed", extras_required=("fastembed",), model_id="fake/v1"
        )
        with patch("mempalace.embeddings.is_installed", return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                cmd_models(args)
    assert exc_info.value.code == 2


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_download_default_slug(mock_config_cls, capsys):
    """The 'default' slug returns None from load_embedding_function — nothing to download."""
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="download", slug="default")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(
            slug="default",
            backend="chroma-default",
            extras_required=(),
            model_id="all-MiniLM-L6-v2",
        )
        with patch("mempalace.embeddings.is_installed", return_value=True):
            with patch("mempalace.embeddings.load_embedding_function", return_value=None):
                cmd_models(args)
    out = capsys.readouterr().out
    assert "chroma default" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_download_failure_exits(mock_config_cls, capsys):
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="download", slug="fake")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(
            slug="fake", backend="fake", extras_required=(), model_id="fake/v1"
        )
        with patch("mempalace.embeddings.is_installed", return_value=True), patch(
            "mempalace.embeddings.load_embedding_function",
            side_effect=RuntimeError("disk full"),
        ), pytest.raises(SystemExit) as exc_info:
            cmd_models(args)
    assert exc_info.value.code == 1


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_models_unknown_action_exits(mock_config_cls, capsys):
    mock_config_cls.return_value = _models_config()
    args = argparse.Namespace(models_action="weird", slug="fake")
    with patch("mempalace.embeddings.get_spec") as mock_spec:
        mock_spec.return_value = MagicMock(slug="fake", backend="fake", extras_required=())
        with pytest.raises(SystemExit) as exc_info:
            cmd_models(args)
    assert exc_info.value.code == 2


# ── cmd_rerankers ─────────────────────────────────────────────────────


def test_cmd_rerankers_list(capsys):
    args = argparse.Namespace(rerankers_action="list")
    cmd_rerankers(args)
    out = capsys.readouterr().out
    assert "SLUG" in out
    assert "PRUNE" in out


def test_cmd_rerankers_default_action(capsys):
    args = argparse.Namespace(rerankers_action=None)
    cmd_rerankers(args)
    out = capsys.readouterr().out
    assert "SLUG" in out


def test_cmd_rerankers_unknown_action(capsys):
    args = argparse.Namespace(rerankers_action="weird")
    with pytest.raises(SystemExit) as exc_info:
        cmd_rerankers(args)
    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    assert "unknown rerankers action" in out


# ── cmd_kg_extract ────────────────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_kg_extract_no_palace(mock_config_cls, tmp_path):
    mock_config_cls.return_value.palace_path = str(tmp_path / "nonexistent")
    args = argparse.Namespace(palace=None, mode="heuristic", model="llama3.1:8b")
    with pytest.raises(SystemExit) as exc_info:
        cmd_kg_extract(args)
    assert exc_info.value.code == 1


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_kg_extract_heuristic(mock_config_cls, tmp_path, capsys):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    mock_config_cls.return_value.palace_path = str(palace_dir)
    args = argparse.Namespace(palace=None, mode="heuristic", model="llama3.1:8b")
    with patch("mempalace.kg_extract.extract_from_palace") as mock_extract:
        mock_extract.return_value = {
            "drawers_scanned": 10,
            "triples_added": 3,
            "errors": 0,
        }
        cmd_kg_extract(args)
    out = capsys.readouterr().out
    assert "Drawers scanned:  10" in out
    assert "Triples added:    3" in out


# ── cmd_mine with --extract-kg flag ─────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_rejects_model_all(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/src",
        palace=None,
        mode="projects",
        wing=None,
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=[],
        extract="exchange",
        model="all",
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_mine(args)
    assert exc_info.value.code == 2


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_with_extract_kg(mock_config_cls, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/src",
        palace=None,
        mode="projects",
        wing=None,
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=[],
        extract="exchange",
        model=None,
        extract_kg=True,
        kg_extract_mode="heuristic",
        kg_extract_model="llama3.1:8b",
    )
    with patch("mempalace.miner.mine"):
        with patch("mempalace.kg_extract.extract_from_palace") as mock_extract:
            mock_extract.return_value = {"drawers_scanned": 2, "triples_added": 5, "errors": 0}
            cmd_mine(args)
    out = capsys.readouterr().out
    assert "Running KG extraction" in out
    assert "added 5 triples" in out


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_mine_extract_kg_failure_degrades(mock_config_cls, capsys):
    mock_config_cls.return_value.palace_path = "/fake/palace"
    args = argparse.Namespace(
        dir="/src",
        palace=None,
        mode="projects",
        wing=None,
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=[],
        extract="exchange",
        model=None,
        extract_kg=True,
        kg_extract_mode="heuristic",
        kg_extract_model="llama3.1:8b",
    )
    with patch("mempalace.miner.mine"):
        with patch("mempalace.kg_extract.extract_from_palace", side_effect=RuntimeError("oops")):
            cmd_mine(args)  # must not raise
    out = capsys.readouterr().out
    assert "KG extraction skipped" in out


# ── cmd_search keyword/keyword_prefix conflict + warm_trie ────────────


def test_cmd_search_rejects_keyword_mix():
    args = _search_args(keyword=["a"], keyword_prefix=["b"])
    with pytest.raises(SystemExit) as exc_info:
        cmd_search(args)
    assert exc_info.value.code == 2


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_search_warm_trie_on_missing_dir(mock_config_cls, tmp_path, capsys):
    """warm_trie=True on a missing trie dir just falls through silently."""
    mock_config_cls.return_value.palace_path = str(tmp_path / "nonexistent")
    args = _search_args(warm_trie=True)
    with patch("mempalace.searcher.search"):
        cmd_search(args)


# ── cmd_status with trie index ────────────────────────────────────────


@patch("mempalace.cli.MempalaceConfig")
def test_cmd_status_with_existing_trie(mock_config_cls, tmp_path):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()
    trie_dir = palace_dir / "trie_index.lmdb"
    trie_dir.mkdir()
    mock_config_cls.return_value.palace_path = str(palace_dir)
    args = argparse.Namespace(palace=None)
    with patch("mempalace.miner.status"):
        with patch("mempalace.trie_index.TrieIndex") as mock_trie_cls:
            mock_trie = MagicMock()
            mock_trie.stats.return_value = {
                "unique_tokens": 10,
                "unique_drawers": 4,
                "postings": 20,
                "db_path": str(trie_dir),
            }
            mock_trie_cls.return_value = mock_trie
            cmd_status(args)
