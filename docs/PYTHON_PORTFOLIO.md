# Hyponoia — Guided Python Portfolio

This page is a compact route through the Python work behind Hyponoia. It does
not ask the reader to inspect every utility, test or support file. The selected
scripts below show the main research and creative process in five stages.

Someone who only wants to use Hyponoia does not need these files. They can
download the latest `_macOS.zip` from
[GitHub Releases](https://github.com/penbek7-maker/hyponoia-generative-composer/releases),
install it once and work through the graphical interface.

## 1. Turning recordings into sound memory

[`memory_builder_v3.py`](../memory_builder_v3.py)

This script analyses each WAV recording, divides it into reusable sound
objects, assigns stable content-derived identities and describes properties
such as energy, brightness, attack, pitch, noisiness, harmonicity, musicality
and possible phrase role.

It demonstrates the first artistic decision of Hyponoia: recordings are not
treated as anonymous files. They become a searchable, reusable musical memory.

## 2. Learning relationships between sounds

[`representation_learning_v1.py`](../representation_learning_v1.py)

This is the compact deep-representation layer. It converts audio fragments to
normalised log-mel representations and defines the encoder used to place sound
objects in a learned embedding space.

The frozen release model is used to discover relationships between sounds. It
does not generate a waveform by itself and it does not copy the accepted
reference compositions.

## 3. Building D1, D3 and D5 compositions

[`generator_v3_memory_bloom_smooth.py`](../generator_v3_memory_bloom_smooth.py)

This is the main composition engine. It selects related material, organises
phrase roles and form, and applies sample transformation, granulation,
synthetic layers, arpeggios, development, transitions and final click-safe
envelopes.

D1, D3 and D5 are not fixed songs. They are different compositional depths
that share a learned aesthetic baseline while retaining level-specific
behaviour.

## 4. Learning from the listener

[`adaptive_composition_preference_v1.py`](../adaptive_composition_preference_v1.py)

This script shows the composition-wide feedback loop. Ratings, the explicit
keep/reject decision and confirmed language feedback become one auditable
review event. The preference layer makes bounded updates that can influence
material choice, activity, richness, synth presence, development, transitions
and form in later generations.

The frozen baseline is preserved, so personal learning can be backed up or
reset rather than silently destroying the original model state.

## 5. Understanding free comments

[`local_llm_feedback_v1.py`](../local_llm_feedback_v1.py)

The language layer uses a small local model to interpret free Greek or English
feedback in context. It is constrained to Hyponoia's known musical intentions,
shows the interpretation before applying it and falls back to a clearly
labelled basic parser when the local model is unavailable.

## Complete process

`Personal WAV library → sound objects → deep embeddings → D1/D3/D5 composition
→ listen → ratings/comment/voice → confirmed preference update → generate again`

## Supporting code

The repository also contains stability, library-update, backup, voice,
analysis, interface and testing modules. They remain part of the reproducible
application, but they are intentionally excluded from this short portfolio
route so the central idea can be understood without navigating the entire
codebase.

The optional Max/MSP connection is implemented by `generator_receiver.py` and
is documented separately because it is an integration component, not one of
the five central Python portfolio works.
