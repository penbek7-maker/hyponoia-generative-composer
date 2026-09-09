# Hyponoia — Programmer Guide

This guide is for a programmer who wants to inspect, test or extend Hyponoia.
End users should use the packaged macOS application from GitHub Releases.

## Supported development environment

- macOS
- Python 3.10, 3.11 or 3.12
- Max 9 only for the optional live-performance connection
- Ollama with the pinned `qwen3:4b` model only for contextual local-language
  understanding; the application keeps a labelled basic fallback

## Set up the source checkout

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-app.txt
```

Run the desktop application with:

```bash
python hyponoia_app.py
```

## Main entry points

- `hyponoia_app.py` — unified graphical application
- `update_library_v1.py` — incremental user-library update
- `generator_v3_memory_bloom_smooth.py` — D1/D3/D5 composition engine
- `hyponoia_feedback_app.py` — ratings, comment and voice workflow
- `adaptive_composition_preference_v1.py` — composition-wide preference model
- `local_llm_feedback_v1.py` — constrained contextual comment interpretation
- `generator_receiver.py` — optional Max/MSP OSC bridge

The five files selected for a shorter code reading are documented in
[`PYTHON_PORTFOLIO.md`](PYTHON_PORTFOLIO.md).

## Runtime data

The release ships the frozen encoder, metric adapter, calibrated embeddings,
gold composition-preference baseline and learning seed in `phase2_artifacts/`.
They are treated as immutable release assets.

User choices, WAV files, generated audio, feedback history and personal
learning are local runtime data. They are ignored by Git and must never be
committed or bundled into a public release.

## Run the tests

```bash
python -m pytest -q
```

The current release gate contains 185 tests. They cover audio continuity,
incremental embeddings, D1/D3/D5 generation behaviour, feedback, contextual
language interpretation, preference learning, backup/reset, packaging and the
Max OSC bridge.

## Build the macOS package

```bash
python build_release_package.py --destination /absolute/output/folder --version VERSION
```

The builder uses an explicit runtime allow-list. Research utilities, test-only
dependencies, caches, old packages and personal data are excluded from the
user ZIP.

## Safety rules for changes

1. Do not overwrite files in `phase2_artifacts/` during personal learning.
2. Keep user feedback append-only and preserve the backup/reset path.
3. Keep all generated and personal data excluded through `.gitignore`.
4. Run the complete test suite after changing paths, imports or packaging.
5. Perform a clean-install test before creating a public GitHub release.
