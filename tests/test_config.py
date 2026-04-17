import json
import os

from mempalace.config import MempalaceConfig


def test_default_config(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert "palace" in cfg.palace_path
    assert cfg.collection_name == "mempalace_drawers"


def test_config_from_file(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"palace_path": "/custom/palace"}))
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path == "/custom/palace"


def test_env_override(tmp_path):
    os.environ["MEMPALACE_PALACE_PATH"] = "/env/palace"
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path == "/env/palace"
    del os.environ["MEMPALACE_PALACE_PATH"]


def test_init(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    cfg.init()
    assert (tmp_path / "config.json").exists()


# ── hook settings round-trip ───────────────────────────────────────────


def test_hook_settings_default_when_unset(tmp_path):
    """Palaces without a hooks block fall back to documented defaults."""
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.hook_silent_save is True
    assert cfg.hook_desktop_toast is False


def test_hook_settings_round_trip(tmp_path):
    """set_hook_setting writes to disk and survives a fresh config load."""
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    cfg.init()  # create config.json so the setter has a file to merge into
    cfg.set_hook_setting("silent_save", False)
    cfg.set_hook_setting("desktop_toast", True)

    # Round-trip via a fresh config object reading from the same dir.
    reloaded = MempalaceConfig(config_dir=str(tmp_path))
    assert reloaded.hook_silent_save is False
    assert reloaded.hook_desktop_toast is True

    # Mutate again to confirm partial updates preserve the other key.
    reloaded.set_hook_setting("silent_save", True)
    final = MempalaceConfig(config_dir=str(tmp_path))
    assert final.hook_silent_save is True
    assert final.hook_desktop_toast is True


def test_hook_settings_preserve_other_config_keys(tmp_path):
    """Writing a hook setting must not clobber unrelated config keys."""
    (tmp_path / "config.json").write_text(json.dumps({"palace_path": "/keep/this"}))
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    cfg.set_hook_setting("silent_save", False)

    reloaded = MempalaceConfig(config_dir=str(tmp_path))
    assert reloaded.palace_path == "/keep/this"
    assert reloaded.hook_silent_save is False
