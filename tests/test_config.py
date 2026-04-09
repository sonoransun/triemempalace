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
