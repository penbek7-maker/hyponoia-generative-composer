# Voice and composition-wide feedback checkpoint — 3 September 2026

## What works now

- The active D1, D3 or D5 feedback question can be read aloud by the local
  macOS Greek voice.
- A visible `Ελληνικά / English` selector changes the complete feedback
  interface, question and local reading voice. Written and spoken input remains
  language-auto-detected, so the interface language does not restrict what the
  listener may say.
- Recording is push-to-talk: it starts and stops only after an explicit user
  click and has a 30-second safety limit.
- Multilingual Whisper `base` transcribes Greek or English locally on CPU.
- The transcript appears in the same comment field and can be corrected before
  interpretation.
- Typed and spoken comments use the same local `qwen3:4b` interpreter.
- Hyponoia validates the model output against a fixed list of musical intents;
  the language model cannot invent controls or numeric changes.
- Nothing changes until the user sees the interpretation and presses Apply.
- The evidence event records the source (`text` or `voice`), language, model,
  bounded control changes and affected composition domains.
- Ratings and comments remain isolated to the selected D1, D3 or D5 profile.

## Composition domains

The current bounded controls can affect:

1. material selection and relationships;
2. synth and arpeggios;
3. energy and density;
4. layers and mix clarity;
5. material development;
6. transitions and sound departures;
7. overall form.

The preview lists only the domains actually reached by the current feedback.
For example, “more synth, more energy and smoother departures” reaches
material selection, synth/arpeggios, energy/density, transitions and overall
form.

## Privacy and safety boundary

Speech-to-text, text interpretation and question reading are local. Hyponoia
does not send microphone audio or comments to a remote API. Recorded samples
exist only in memory until transcription finishes. Low-confidence or invalid
language-model output cannot be applied, and technical model failure falls
back to the smaller deterministic interpreter.

## Research boundary

This checkpoint makes listener feedback composition-wide in the sense that it
can influence all major generator domains through explicit bounded controls.
It is not yet the later learned composition-wide preference model. That model
will learn from complete-render representations plus ratings, comments and the
keep/reject decision. The current evidence format is designed to supply its
training examples without erasing the explainable control path.

## Verification

- Full suite: 107 tests passed.
- Live local-Qwen Greek phrase: interpreted and previewable.
- Live local-Qwen combined synth/energy/transition phrase: interpreted and
  routed to the expected domains.
- Microphone permission, real speech transcription and audible question are
  intentionally verified by the user through the application UI.
