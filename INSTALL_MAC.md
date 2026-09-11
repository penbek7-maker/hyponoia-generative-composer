# Hyponoia — macOS Installation

## Simple installation

For normal use there are only two steps:

1. Double-click `Install Hyponoia.command` once.
2. Double-click `Open Hyponoia.command` whenever you want to compose.

The first time you open **Teach Hyponoia**, press **Enable context
understanding**. If the local Ollama runtime is missing, Hyponoia opens its
official macOS download page. Install and open Ollama once, return to Hyponoia
and press the same button again. Hyponoia then downloads the pinned multilingual
`qwen3:4b` model (about 2.5 GB). This is a one-time step; comments remain on
the user's computer. Until it is complete, Hyponoia clearly reports **BASIC**
and uses its limited safety vocabulary instead of claiming contextual
understanding.

The installer creates an isolated Python environment and prepares the local
Greek/English voice model. It never uploads, moves or edits the user's WAV
library. If macOS blocks a downloaded `.command` file, right-click it once and
choose **Open**.

Inside Hyponoia, follow the four numbered tabs:

- **Library** — choose any folder of WAV files; around 100 is recommended, not required.
- **Generate & Listen** — create D1, D3 or D5 and listen to the latest render.
- **Teach Hyponoia** — use ratings, text or local voice feedback.
- **Live / Max** — optionally start the OSC connection for live performance.

The manual steps below remain available for diagnosis and research use.

## 1. Keep the old project untouched

Extract this package as a new folder, for example:

`~/Desktop/hyponoia-generative-composer`

Keep any previous project as a recovery copy until installation, private-data
validation and the Max live round trip have all passed.

## 2. Check Python

Open Terminal and run:

```bash
python3 --version
```

Use Python 3.10, 3.11, or 3.12 for this pinned baseline. If Terminal reports another version or says that Python is missing, stop before continuing.

## 3. Create the isolated environment manually (developers)

The downloaded macOS release should normally be installed with
`Install Hyponoia.command`. The commands below are for developers working from
a complete GitHub checkout, which also contains the test suite and development
requirements.

```bash
cd ~/Desktop/hyponoia-generative-composer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

When the environment is active, Terminal shows `(.venv)` at the beginning of the line.

## 4. Run the code tests

```bash
python -m pytest -q
```

Expected result for the current release branch: `213 passed`.

## 5. Transfer personal Hyponoia data

Copy only the following from the old project into the new folder:

- all personal WAV files from old `alpha_memory/` into new `alpha_memory/`;
- `learning_profile.json`, if previous global learning should be retained;
- `sample_learning_profile.json`, if previous sample-level learning should be migrated.

Do not copy or replace:

- `memory_index_v3.json` — rebuild it for the 48 kHz/stable-ID baseline;
- old Python scripts — the package already contains the corrected versions;
- `output/`, `render_reports/`, or `critic_reports/` — keep these in the old folder as archive;
- `.venv/` — it belongs only to the computer/environment where it was created.

## 6. Rebuild the sound memory

With `(.venv)` active:

```bash
python memory_builder_v3.py
```

Successful output reports the number of recordings and sound objects and creates:

- `memory_index_v3.json`
- `memory_build_report.json`

## 7. Build the musical profile

Choose one representative WAV from `alpha_memory/` and run, replacing `REFERENCE.wav` with its exact filename:

```bash
python build_alpha_profile.py "alpha_memory/REFERENCE.wav" --output alpha_profile.json
```

Quotation marks are important when the filename contains spaces.

## 8. Generate the first composition

Free/non-tonal D5 render:

```bash
python generator_v3_memory_bloom_smooth.py 5 0 free 0
```

C minor D5 render with confident harmonic guidance:

```bash
python generator_v3_memory_bloom_smooth.py 5 0 minor 0.85
```

The three-minute stereo 48 kHz result is saved in `output/`. The most recent version is also `output/current.wav`.

## 9. Run the Critic

Replace `RENDER_NAME.wav` with the timestamped generated filename:

```bash
python critic_v2.py "output/RENDER_NAME.wav"
```

## 10. Give ratings, text or voice feedback

Replace `REPORT_NAME_critic.json` with the generated Critic report:

```bash
python human_feedback_v1.py "critic_reports/REPORT_NAME_critic.json"
```

Normal users should use the **Teach Hyponoia** tab. It combines seven ratings
from 1 to 5, a keep/reject/unsure decision and free Greek or English input.
Voice is transcribed locally by multilingual Whisper and then follows the same
preview-and-confirm path as typed text.

The older command-line feedback tool remains available for research workflows.
Its controlled terms include:

- `more library objects`
- `greater exploration`
- `less repetition`
- `more synthesizers`
- `more musical`
- `more rhythmic`
- `greater bloom`
- `more energetic`
- `faster`
- `smoother transitions`

## 11. Regenerate and verify learning

Run the generator again with the same D-level/harmony command. The console must now print the updated learning weights instead of neutral values.

## Important recovery rule

Until Gate 1 passes with the real personal library and the Max live round trip,
do not permanently delete the recovery copy. If any command fails, stop and
preserve the complete Terminal output.
