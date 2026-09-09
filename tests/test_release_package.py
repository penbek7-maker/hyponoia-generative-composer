from pathlib import Path

from build_release_package import is_engine_source


def test_release_engine_includes_runtime_and_excludes_development_clutter():
    assert is_engine_source(Path("hyponoia_app.py"))
    assert is_engine_source(Path("requirements-app.txt"))
    assert not is_engine_source(Path("requirements-dev.txt"))
    assert not is_engine_source(Path("critic_recalibration_v21.py"))
    assert not is_engine_source(Path("build_release_package.py"))
    assert is_engine_source(Path("phase2_artifacts/embeddings_v2.json"))
    assert is_engine_source(Path("alpha_memory/.gitkeep"))
    assert not is_engine_source(Path("tests/test_audio_pipeline.py"))
    assert not is_engine_source(Path("Hyponoia_Phase2_User_Package_2026-09-01.zip"))
    assert not is_engine_source(Path("Screenshot 2026-07-11 at 15.33.33.png"))
