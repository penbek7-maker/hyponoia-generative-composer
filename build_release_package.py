"""Build a clean combined macOS release without personal audio or learning data."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_VERSION = "2.0.0-rc3"
RUNTIME_JSON = {"alpha_profile.json", "representation_config.json", "library_source_labels.json"}
RUNTIME_REQUIREMENTS = {
    "requirements.txt",
    "requirements-app.txt",
    "requirements-representation.txt",
    "requirements-voice.txt",
}
RUNTIME_PYTHON = {
    "adaptive_composition_preference_v1.py",
    "artist_style_v1.py",
    "composition_feedback_v1.py",
    "composition_influence_v1.py",
    "composition_preference_v1.py",
    "feedback_input_v1.py",
    "generator_receiver.py",
    "generator_v3_memory_bloom_smooth.py",
    "human_feedback_v1.py",
    "hyponoia_app.py",
    "hyponoia_feedback_app.py",
    "hyponoia_runtime.py",
    "hyponoia_stability.py",
    "incremental_embeddings_v1.py",
    "install_voice_model_v1.py",
    "learning_backup_v1.py",
    "learning_profile_store_v1.py",
    "library_coverage_v1.py",
    "library_manager_v1.py",
    "local_language_model_v1.py",
    "local_llm_feedback_v1.py",
    "max_live_v1.py",
    "memory_builder_v3.py",
    "representation_assist_v1.py",
    "representation_feedback_v2.py",
    "representation_learning_v1.py",
    "representation_training_v1.py",
    "update_library_v1.py",
    "voice_feedback_v1.py",
}


def is_engine_source(relative: Path) -> bool:
    if len(relative.parts) == 1:
        return (
            relative.name in RUNTIME_PYTHON
            or relative.name in RUNTIME_REQUIREMENTS
            or relative.name in RUNTIME_JSON
        )
    return (
        relative.parts[0] == "phase2_artifacts"
        or relative == Path("alpha_memory/.gitkeep")
        or relative == Path("assets/hyponoia_python_bg.png")
    )


def tracked_paths(source: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=source)
    return [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]


def _copy_engine(source: Path, engine: Path, paths: list[Path]) -> None:
    for relative in paths:
        if not is_engine_source(relative):
            continue
        origin = source / relative
        if not origin.is_file():
            continue
        destination = engine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)


def _release_launcher(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        'cd "$SCRIPT_DIR"',
        'ENGINE_DIR="$SCRIPT_DIR/Hyponoia Engine"\ncd "$ENGINE_DIR"',
        1,
    )
    destination.write_text(text, encoding="utf-8")
    destination.chmod(0o755)


def build_release(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    max_project_dir: str | Path | None = None,
    version: str = DEFAULT_VERSION,
    source_paths: list[Path] | None = None,
) -> dict:
    source = Path(source_dir).resolve()
    destination = Path(destination_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    package_name = f"Hyponoia_{version}_macOS"
    archive_path = destination / f"{package_name}.zip"
    if archive_path.exists():
        raise FileExistsError(f"Release already exists: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="hyponoia-release-") as temporary:
        release = Path(temporary) / package_name
        engine = release / "Hyponoia Engine"
        engine.mkdir(parents=True)
        _copy_engine(source, engine, source_paths if source_paths is not None else tracked_paths(source))
        _release_launcher(source / "Install Hyponoia.command", release / "Install Hyponoia.command")
        _release_launcher(source / "Open Hyponoia.command", release / "Open Hyponoia.command")
        shutil.copy2(source / "INSTALL_MAC.md", release / "START HERE.md")
        shutil.copy2(source / "README.md", release / "TECHNICAL README.md")

        max_included = False
        if max_project_dir is not None:
            max_source = Path(max_project_dir).expanduser().resolve()
            if not max_source.is_dir():
                raise FileNotFoundError(f"Max project folder not found: {max_source}")
            shutil.copytree(
                max_source,
                release / "Max Live" / max_source.name,
                ignore=shutil.ignore_patterns(".git", "*.zip", ".DS_Store"),
            )
            max_included = True

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        manifest = {
            "product": "Hyponoia",
            "version": version,
            "git_commit": commit,
            "max_project_included": max_included,
            "artist_style_baseline_included": (
                engine / "phase2_artifacts" / "artist_style_baseline_v1.json"
            ).is_file(),
            "personal_data_included": False,
            "installation": "Double-click Install Hyponoia.command once, then Open Hyponoia.command",
        }
        (release / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.make_archive(str(archive_path.with_suffix("")), "zip", Path(temporary), package_name)
    return {"archive": str(archive_path), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--max-project", type=Path)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    result = build_release(
        Path(__file__).resolve().parent,
        args.destination,
        max_project_dir=args.max_project,
        version=args.version,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
