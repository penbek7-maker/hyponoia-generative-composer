# Hyponoia Generative Composer

Hyponoia is an AI-assisted generative music system for adaptive soundscape composition using sample memory, scale-aware material selection, OSC communication with Max/MSP, internal audio analysis, and human-guided learning. The stabilised baseline uses a consistent 48 kHz workflow and content-derived recording/object IDs.

The system selects and transforms musical materials from a user-defined audio memory. Human evaluation and an internal critic progressively influence global compositional weights and sample-level selection values.

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

## 7. Use Hyponoia With Max/MSP

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

`/generator/ready 1`

`/generator/error error_code`

The path is sent before the ready trigger so Max can preload the completed WAV
without a race condition.

The complete Max → Python render → dynamic WAV path → Max playback round trip
was verified live on 21 August 2026 while the physiological input remained on
port `5001`.

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

Human ratings are the primary learning target. The Critic is stored as an auxiliary diagnostic signal rather than silently overriding the user's evaluation. Numeric ratings update the common aesthetic profile. Optional English comments use a small deterministic MVP vocabulary; for example, `more musical`, `more rhythmic`, `greater bloom`, `more library objects`, `greater exploration`, `less repetition`, `more synthesizers`, `more energetic`, `faster`, `smoother transitions`, `fewer samples`, `develop the selected sounds`, and `use different materials` change bounded generator controls and are logged with before/after values. Text intent is stored separately for D1, D3 and D5, inferred from the rated render or an explicit `D1:`, `D3:` or `D5:` prefix. All three profiles change only after explicit global language such as `globally` or `in all levels`.

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

D5 activity is deliberately not interpreted as unlimited layer count. The
reference-continuity revision organises foreground events around a soft shared
122–129 BPM pulse, maintains overlap between related role lanes, and guarantees
cosine-shaped attacks and releases after all other transformations. This keeps
energy and synthetic motion while preventing abrupt, scissor-like phrase cuts.

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
This is currently an explainable adaptive-learning layer; deep representation
learning is intentionally reserved for the next research phase.

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

Hyponoia investigates adaptive and affective approaches to computer-assisted musical composition.

Rather than directly mapping biosignals to isolated synthesis parameters, the broader research framework explores the interpretation of higher-level computational states as compositional behaviours.

The generative engine presented here forms part of an ongoing research system for adaptive musical performance, biofeedback interaction, and human-guided machine learning.

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
