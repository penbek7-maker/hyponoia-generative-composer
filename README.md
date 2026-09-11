# Hyponoia Generative Composer

Hyponoia is a local, AI-assisted generative composition system. It creates and
develops music from a sound library selected by the user, then learns from
ratings, written comments and voice feedback.

## Download and play — no coding required

**[Download Hyponoia for macOS from GitHub Releases](https://github.com/penbek7-maker/hyponoia-generative-composer/releases)**

Open the newest release, expand **Assets**, and download the file whose name
ends in `_macOS.zip`. Do not download the automatic **Source code** archives.

After extracting the ZIP:

1. Double-click `Install Hyponoia.command` once.
2. Double-click `Open Hyponoia.command` whenever you want to use Hyponoia.
3. Choose a folder containing your WAV sounds.
4. Generate, listen, give feedback and generate again.

Python commands are not required for this path. Max/MSP is optional and is used
only for the extended live-performance environment.

## Explore how it was made

The public project is organised as a readable research portfolio as well as a
downloadable musical tool:

- **[Guided Python portfolio](docs/PYTHON_PORTFOLIO.md):** a short path through
  the selected scripts that show how the system was built, from sound memory
  and embeddings to generation and preference learning.
- **[Programmer guide](docs/PROGRAMMER_GUIDE.md):** the exact development
  environment, entry points, tests, private-data boundaries and release build.
- **Application and learning system:** the remaining source stays available
  for reproducibility, but the portfolio deliberately highlights only the
  files that explain the main creative and research decisions.
- **Max/MSP:** the optional live-performance project is kept separately in the
  [Hyponoia Max/MSP repository](https://github.com/penbek7-maker/hyponoia-maxmsp).

The final pre-release checks are tracked in the
**[release checklist](docs/RELEASE_CHECKLIST.md)**.

## What Hyponoia does

Hyponoia uses sample memory, scale-aware material selection, OSC communication
with Max/MSP, internal audio analysis, deep representation assistance and
human-guided learning. The stabilised baseline uses a consistent 48 kHz
workflow and content-derived recording/object IDs.

The system selects and transforms musical materials from a user-defined audio memory. Human evaluation and an internal critic progressively influence global compositional weights and sample-level selection values.

## Desktop quick start (macOS)

1. Double-click `Install Hyponoia.command` once.
2. Double-click `Open Hyponoia.command`.
3. In **Library**, choose a folder containing WAV files and press **Update library**.
4. In **Generate & Listen**, choose D1, D3 or D5 and press **Generate**.
5. Listen, then use **Feedback** to write or speak naturally in Greek or English.
6. Generate again to hear the bounded learned changes.

Around 100 recordings are recommended, not required. The user can add, remove,
replace or rename WAV files later. Hyponoia reuses unchanged analyses and deep
embeddings, creates embeddings only for new/changed sound objects, and removes
inactive embeddings without a complete retraining run.

The repository includes the frozen 32-dimensional representation model, 2,566
calibrated sound-object embeddings, the locked D1/D3/D5 gold preference head and
the approved bounded artist-style baseline. The gold head guides material and
structure without copying the three reference waveforms or banning the rest of
a user's library. The artist baseline stores only an aggregate prototype,
descriptors and twelve-point musical trajectories learned from five authorised
works; no source recording is included.

## 1. Requirements

* Python 3.10 or newer
* Max/MSP for real-time OSC control
* WAV audio samples

## 2. Installation

Open Terminal and move into the project folder.

Example:

`cd ~/Desktop/hyponoia-generative-composer`

Create a Python virtual environment:

`python3 -m venv .venv`

Activate it on macOS or Linux:

`source .venv/bin/activate`

Install the required Python packages:

`pip install -r requirements.txt`

## 3. Add Your Own Audio Memory

Place your own WAV audio files inside:

`alpha_memory/`

Example:

`alpha_memory/sample1.wav`

`alpha_memory/sample2.wav`

`alpha_memory/sample3.wav`

The audio memory is intentionally user-defined. Each composer can populate the folder with their own recordings, instrumental materials, field recordings, electronic sounds, or other source material.

Personal WAV files are ignored by Git and are not uploaded to the repository.

## 4. Build the Sample Memory

Run:

`python3 memory_builder_v3.py`

The system analyses the audio material and creates:

`memory_index_v3.json`

## 5. Build the Musical Profile

Run:

`python3 build_alpha_profile.py`

This creates or updates:

`alpha_profile.json`

## 6. Generate a Soundscape Without Max/MSP

For a free harmonic render:

`python3 generator_v3_memory_bloom_smooth.py 5 0 free 0`

For a scale-aware render in C minor:

`python3 generator_v3_memory_bloom_smooth.py 5 0 minor 0.85`

Dream levels are:

`1` = simple

`3` = medium

`5` = rich

Root pitch classes are:

`0 = C`

`1 = C#`

`2 = D`

`3 = D#`

`4 = E`

`5 = F`

`6 = F#`

`7 = G`

`8 = G#`

`9 = A`

`10 = A#`

`11 = B`

Generated audio is stored inside:

`output/`

The latest render is also available as:

`output/current.wav`

Each generation also saves three remix-ready, complementary 32-bit float WAVs:
`LOW` (below 250 Hz), `MID` (250 Hz–4 kHz), and `HIGH` (above 4 kHz).
They can be loaded into a DAW or Max/MSP and recombined to reconstruct the unchanged
master, or balanced separately for a new mix.

## 7. Use Hyponoia With Max/MSP

Normal users open the **Live / Max** tab and press **Start live connection**.
The command-line details below remain useful for debugging and custom patches.

Start the OSC receiver:

`python3 generator_receiver.py`

The Python receiver listens on:

`127.0.0.1:7401`

Max/MSP can send the following OSC messages:

`/harmony/root 0`

`/harmony/scale minor`

`/harmony/confidence 0.85`

`/generator/render D5`

Python sends messages back to Max/MSP on port `7402`.

Returned OSC messages are:

`/generator/path /absolute/path/to/current.wav`

`/generator/path/low /absolute/path/to/current_LOW.wav`

`/generator/path/mid /absolute/path/to/current_MID.wav`

`/generator/path/high /absolute/path/to/current_HIGH.wav`

`/generator/ready 1`

`/generator/error error_code`

The path is sent before the ready trigger so Max can preload the completed WAV
without a race condition.

The complete Max → Python render → dynamic WAV path → Max playback round trip
was verified again with the release candidate on 11 September 2026 while the
physiological input remained independent on port `5001`.

The generator's scale-aware selection remains in free mode when harmonic confidence is below `0.55`.

## 8. Run the Internal Critic

After generating a WAV file, run:

`python3 critic_v2.py output/YOUR_RENDER_NAME.wav`

Critic v2.1 analyses musicality, coherence, richness, transitions, and bloom
development. Coherence combines local continuity, macro-form motion,
development and material balance instead of rewarding smoothness alone. It
also reports structural similarity to earlier renders at the same D-level.

It also evaluates sample diversity, repetition, and exploration using the corresponding render report.

Critic reports are stored inside:

`critic_reports/`

## 9. Give Human Feedback

Run:

`python3 human_feedback_v1.py critic_reports/YOUR_RENDER_NAME_critic.json`

Enter scores from 0 to 100 for the requested musical criteria.

Before rating, read [`docs/USER_FEEDBACK_GUIDE.md`](docs/USER_FEEDBACK_GUIDE.md). The short rule is: rate perceived quality, intentionality, and development—not the number of events or agreement with the Critic. The same criteria apply to D1, D3, and D5; a sparse D1 can still receive high richness when its limited material evolves meaningfully.

The feedback process updates:

`learning_profile.json`

and:

`sample_learning_profile.json`

Human ratings are the primary learning target. The Critic is stored as an
auxiliary diagnostic signal rather than silently overriding the user's
evaluation. Free Greek or English comments use the pinned local `qwen3:4b`
language model through Ollama, with a clearly labelled deterministic fallback
when that one-time local setup is not ready. Typed and locally transcribed voice input
follow the same preview-and-confirm path. Ratings, language feedback and the
explicit keep/reject/unsure decision form one auditable whole-composition
event. D1, D3 and D5 retain level-specific learning, while confirmed positive
and contrast examples update a bounded composition-wide preference head over
the deep audio embeddings. The frozen release baseline is never overwritten.

## 10. Learning Loop

The complete learning process is:

Generate → Internal Critic → Human Rating/Text Feedback → Explainable Control Update → Exposure/Role-Aware Sample Credit → Next Generation

The generator combines:

* scale-aware material selection
* shared numeric-rating weights plus an isolated D1/D3/D5 text profile
* learned values for individual sample objects
* exploration bonuses for less-used material
* append-only feedback events with unique IDs
* uncertainty and signed credit evidence for each sample update
* bounded D5 energy/musicality drive and three alternative D5 formal arcs
* D5-only temporal drive that shortens stretches, envelopes and delay spacing
* role-aware internal motion that develops the selected D5 material
* reference-derived soft pulse, phrase-lane overlap and a final continuity guard
* source-derived granular beds built from the material selected for each render
* separate source-derived drone, pad and motif-phrase layers with no oscillators
* an approved bounded artist-style baseline for phrase persistence, recurrence,
  granulation, energy/density development and spatial motion

D5 activity is deliberately not interpreted as unlimited layer count. The
reference-continuity revision organises foreground events around a soft shared
122–129 BPM pulse and maintains overlap between related role lanes. Every dream
level now receives cosine-shaped attacks and releases after all other
transformations. This keeps energy and synthetic motion while preventing abrupt,
scissor-like phrase cuts.

The background is not a permanent high- or low-frequency oscillator. Quiet beds
are granular overlap-add re-samplings of the source events selected for that
render. Musical foreground phrases also come from motifs cut from that material,
then developed through scale-related resampling, stretching, granular variation,
overlap and long windows. The dry source mix remains intact, so this change does
not globally remove high frequencies.

Sustained texture/resonance events feed a dedicated drone bus. Events whose
features indicate synth-like material feed a separate pad bus. These buses bloom
through different long-form curves, while a third motif layer supplies rhythmic
and arpeggio-like source phrases. D1, D3 and D5 use increasingly active but still
bounded forms, so intensity comes from development rather than a shared preset.

The aesthetic-bridge revision uses the listener-preferred 20 August D5 as its
primary aesthetic baseline. It restores a broader material palette, elastic
phrase spacing, long envelopes and ambient spectral body, while retaining the
corrected Critic, level-specific feedback, bounded synthetic preference and the
final continuity guard. Pulse attraction and internal modulation are deliberately
subtle: they provide energy without turning the result into a rigid or muddy
stack of repeatedly recycled midrange material.

The system does not use hard sample bans or fixed repetition caps. Previously successful material can return as musical memory, while underexplored material remains available for future selection.

Material selection is feature-driven rather than filename-driven. Candidate
objects are compared through energy, brightness, noisiness, attack, duration,
musicality, richness, harmonicity, resonance, synthetic score, pitch/scale fit,
phrase role and continuity with the preceding object. Human sample values then
modify the weighted probability, while recording-history penalties, within-render
usage penalties and an exploration bonus keep less-used material available.
This combines an explainable adaptive-learning layer with bounded deep
representation assistance and a level-specific composition preference head.
It is not presented as an end-to-end neural composer or as a generalisation
claim from three accepted compositions.

## Main Files

`generator_v3_memory_bloom_smooth.py`

Main generative composition engine.

`generator_receiver.py`

OSC communication between Max/MSP and the Python generator.

`critic_v2.py`

Internal audio and sample-usage critic.

`human_feedback_v1.py`

Human feedback and learning update system.

`memory_builder_v3.py`

Audio memory analysis and object extraction.

`build_alpha_profile.py`

Builds a configurable musical feature profile from a supplied reference WAV.

`hyponoia_stability.py`

Shared stable IDs, atomic persistence, deterministic grouping, migration, and phrase-to-action helpers.

`critic_calibration.py`

Summarises Critic variance, MAE, signed error, and rank correlation against collected human ratings.

`alpha_memory/`

User-defined audio memory folder.

## Research Context

External listener feedback is tracked as release evidence. The first outside
review identified a recurring narrow-band, tinnitus-like signature, a repeated
rise/fall gesture near three quarters of the form, and insufficiently explicit
pitch control. The generator now derives its spectral bloom from the selected
sound library instead of adding a fixed pure-tone field, varies the bloom window
per render, replaces the fixed low bed with a source-derived granular layer, and
exposes root pitch with musical note names. The longer-term
acousmatic-corpus protocol is documented in
[`docs/ACOUSMATIC_CORPUS_PROTOCOL.md`](docs/ACOUSMATIC_CORPUS_PROTOCOL.md).

Hyponoia investigates adaptive and affective approaches to computer-assisted musical composition.

Rather than directly mapping biosignals to isolated synthesis parameters, the broader research framework explores the interpretation of higher-level computational states as compositional behaviours.

The generative engine presented here forms part of an ongoing research system for adaptive musical performance, biofeedback interaction, and human-guided machine learning.

## Development transparency and acknowledgement

The development of the system was assisted by OpenAI Codex during
implementation, code review and the technical integration of the deep-learning
process. The research methodology, data curation, evaluation of results,
creation and validation of the code, and final design and artistic decisions
were carried out by the researcher, with the support of this tool.

## Status

Research prototype and work in progress.

## Regression Tests

Install the development dependencies and run:

`pip install -r requirements-dev.txt`

`pytest -q`

The test suite covers the 48 kHz path, content IDs, v1 profile migration,
feedback direction, sample-selection probability, D1/D3/D5 comment isolation,
append-only history, differentiated sample credit, audio caching, Critic score
variation and non-ceiling coherence, historical structural novelty,
dimension-specific calibration, and non-blocking OSC behaviour.

For the read-only validation of a private real-data project, run:

`python3 gate1_private_validation.py --project-dir .`

This writes `reports/gate1_private_baseline.json` without changing WAV files or
learning profiles.
