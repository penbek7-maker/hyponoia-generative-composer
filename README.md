# Hyponoia Generative Composer

Hyponoia is an AI-assisted generative music system for adaptive soundscape composition using sample memory, scale-aware material selection, OSC communication with Max/MSP, internal audio analysis, and human-guided learning.

The system selects and transforms musical materials from a user-defined audio memory. Human evaluation and an internal critic progressively influence global compositional weights and sample-level selection values.

## 1. Requirements

* Python 3.10 or newer
* Max/MSP for real-time OSC control
* WAV audio samples

## 2. Installation

Open Terminal and move into the project folder.

Example:

`cd ~/Desktop/hyponoia_generator`

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

`/generator/ready 1`

`/generator/path /absolute/path/to/current.wav`

`/generator/error error_code`

The generator's scale-aware selection remains in free mode when harmonic confidence is below `0.55`.

## 8. Run the Internal Critic

After generating a WAV file, run:

`python3 critic_v2.py output/YOUR_RENDER_NAME.wav`

The critic analyses musicality, coherence, richness, transitions, and bloom development.

It also evaluates sample diversity, repetition, and exploration using the corresponding render report.

Critic reports are stored inside:

`critic_reports/`

## 9. Give Human Feedback

Run:

`python3 human_feedback_v1.py critic_reports/YOUR_RENDER_NAME_critic.json`

Enter scores from 0 to 100 for the requested musical criteria.

The feedback process updates:

`learning_profile.json`

and:

`sample_learning_profile.json`

## 10. Learning Loop

The complete learning process is:

Generate → Internal Critic → Human Feedback → Global Learning Update → Sample-Level Learning Update → Next Generation

The generator combines:

* scale-aware material selection
* global learned compositional weights
* learned values for individual sample objects
* exploration bonuses for less-used material

The system does not use hard sample bans or fixed repetition caps. Previously successful material can return as musical memory, while underexplored material remains available for future selection.

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

Builds the musical feature profile used by the generator.

`alpha_memory/`

User-defined audio memory folder.

## Research Context

Hyponoia investigates adaptive and affective approaches to computer-assisted musical composition.

Rather than directly mapping biosignals to isolated synthesis parameters, the broader research framework explores the interpretation of higher-level computational states as compositional behaviours.

The generative engine presented here forms part of an ongoing research system for adaptive musical performance, biofeedback interaction, and human-guided machine learning.

## Status

Research prototype and work in progress.
