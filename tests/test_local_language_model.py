from pathlib import Path

import pytest

from local_language_model_v1 import (
    DEFAULT_MODEL,
    LanguageModelSetupError,
    language_model_status,
)


def executable(tmp_path: Path) -> Path:
    path = tmp_path / "ollama"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_release_uses_small_pinned_multilingual_model():
    assert DEFAULT_MODEL == "qwen3:4b"


def test_status_reports_context_ready_only_when_service_has_the_model(tmp_path):
    cli = executable(tmp_path)

    def read_json(_url, _timeout):
        return {"models": [{"name": "qwen3:4b"}]}

    status = language_model_status(candidates=(cli,), read_json=read_json)
    assert status["runner_installed"] is True
    assert status["service_online"] is True
    assert status["model_installed"] is True
    assert status["ready"] is True


def test_status_keeps_basic_fallback_when_local_service_is_offline(tmp_path):
    cli = executable(tmp_path)

    def offline(_url, _timeout):
        raise LanguageModelSetupError("offline")

    status = language_model_status(candidates=(cli,), read_json=offline)
    assert status["runner_installed"] is True
    assert status["service_online"] is False
    assert status["ready"] is False
