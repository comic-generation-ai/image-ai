import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config.settings as settings_module


def test_hf_token_loaded_from_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=from_test_env\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    settings_module.load_environment_file(env_path)

    assert os.environ.get("HF_TOKEN") == "from_test_env"
