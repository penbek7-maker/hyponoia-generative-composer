# Free feedback foundation — 2 September 2026

Hyponoia now has one safe input contract for typed feedback and future voice
transcripts. Both sources pass through the same local Greek/English
musical-intent interpreter. A listener sees a preview of the understood
intent, target D-level and exact bounded control changes before anything is
stored or applied.

The primary interpreter is the compact local `qwen3:4b` model through Ollama.
It runs without a cloud API and returns a strict JSON structure containing
only an explanation, confidence, ambiguities and allowed intent identifiers.
It cannot write generator parameters. Hyponoia validates that result, routes
the feedback to D1/D3/D5 and maps intents to its own small bounded changes. If
the local service is unavailable, the previous deterministic interpreter is a
safe fallback rather than a reason to lose feedback.

## Safety and research boundary

- No unrecognised sentence is silently applied.
- A low-confidence language-model answer cannot be applied.
- Unknown model intents are rejected instead of reaching the generator.
- No profile changes during preview.
- Applying feedback requires explicit confirmation.
- D1, D3 and D5 remain isolated unless the listener explicitly says that the
  instruction is global.
- The original text or transcript and the exact applied changes are stored as
  append-only evidence.
- Voice transcription is treated only as another input source; it does not
  receive a separate or hidden interpretation path.

## Connection to composition-wide learning

The original comment or voice transcript, the accepted interpretation, the
target D-level and the actual bounded updates are stored together. This is
useful labelled evidence for the later composition-wide preference model. The
language model is an input interpreter; it is not itself the audio composition
model and it does not train on the user's private audio.

## Current boundary

Typed free comments are connected. Microphone capture and local speech-to-text
are not connected yet. That next layer will only create a transcript and feed
it through the same preview, confirmation, level isolation and bounded-update
route.
