# Acousmatic Corpus Protocol

This protocol turns the external request for broader acousmatic learning into a
reproducible research stage. It is not permission to scrape or copy commercial
recordings.

## 1. Corpus eligibility

Include only audio that is public domain, openly licensed for the intended use,
supplied directly with written permission, or created for the study. Record for
every work: composer, title, year, source, licence, permission, checksum, and any
required attribution. Exclude works with uncertain rights.

## 2. Representation

The frozen audio encoder produces segment embeddings. Composition-wide training
uses aggregate trajectories over time, structural measurements, and documented
listener judgements. It must learn relationships such as density, spectral
development, continuity, contrast, and formal pacing—not reproduce a named
composer or memorise a recording.

## 3. Experimental split

Use composer-disjoint training, validation, and test sets. No composer, work, or
near-duplicate excerpt may cross a split. Keep the listener-created Hyponoia
references as a separate artistic anchor rather than silently mixing them into
the external corpus.

## 4. Evaluation

Report retrieval and preference metrics, held-out composer performance,
nearest-neighbour checks for memorisation, and blinded musician listening tests.
Compare the trained model with the frozen Hyponoia baseline under identical
libraries and seeds. Document both improvements and regressions.

## 5. Product boundary

The public release ships a frozen, documented model and an independent personal
preference head. A user's ratings, comments, and voice feedback update only that
private bounded head; they do not silently retrain the shared model or upload
audio.
