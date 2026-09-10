# Hyponoia — Final Release Checklist

## Automated gate

- [x] Complete Python test suite: 186 passed
- [x] Release ZIP excludes personal audio and learning data
- [x] Release ZIP excludes developer-only dependencies and utilities
- [x] Contextual Greek/English feedback has a labelled fallback
- [x] Frozen representation and gold preference baseline remain immutable

## Final user tests

- [x] Add genuinely new WAVs and confirm that new embeddings are created
- [x] Remove a WAV and confirm that inactive objects/embeddings are pruned
- [x] Isolate composition-wide output learning and confirm that it changes the next same-seed render
- [ ] Generate and listen to D1, D3 and D5
- [ ] Apply ratings and a free comment, then generate again
- [ ] Close and reopen Hyponoia and confirm that personal learning persists
- [ ] Test voice transcription and spoken read-back
- [ ] Complete the Max request → render → returned path → playback round trip

## Publication gate

- [ ] Build the final macOS ZIP from the tested commit
- [ ] Install the ZIP as a clean user
- [ ] Confirm the README download link and installation instructions
- [ ] Publish the release asset and checksum
- [ ] Merge the tested cleanup commit into `main`
