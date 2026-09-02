# Free feedback foundation — 2 September 2026

Hyponoia now has one safe input contract for typed feedback and future voice
transcripts. Both sources pass through the same deterministic Greek/English
musical-intent interpreter. A listener sees a preview of the understood
intent, target D-level and exact bounded control changes before anything is
stored or applied.

## Safety and research boundary

- No unrecognised sentence is silently applied.
- No profile changes during preview.
- Applying feedback requires explicit confirmation.
- D1, D3 and D5 remain isolated unless the listener explicitly says that the
  instruction is global.
- The original text or transcript and the exact applied changes are stored as
  append-only evidence.
- Voice transcription is treated only as another input source; it does not
  receive a separate or hidden interpretation path.

The microphone capture UI is the next layer. This commit establishes and tests
the shared text/voice data contract first, so voice cannot bypass preview,
confirmation, level isolation or bounded updates.
