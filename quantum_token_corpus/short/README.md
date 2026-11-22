# Short controlled pronoun corpus

All 1,656 inputs contain **9–15 words**, counting contractions as one word and
excluding punctuation. This count is for the explicit word tokenizer in the
controlled experiment, not a pretrained model's subword tokenizer.

| Tier | Word range | Example |
|---|---|---|
| 1: size | 9–11 | Red bottle won't fit inside blue cooler; it's too large. |
| 2: relation and size | 9–13 | Red poster cannot cover blue noticeboard; it's too small. |
| 3: comparison, update, query | 13–15 | Silver screen begins brighter than gold screen. Settings unchanged; select it, now brighter. |

The original corpus's IDs, labels, candidate order, scenario assignments, split
sizes, and starter subsets are preserved. Only the sentences are rewritten.
Tier 3 uses concise clauses and still includes every required state update.
The original files remain in the parent directory.

Regenerate from the repository root:

```bash
python3 quantum_token_corpus/make_short_corpus.py
```

Run both quantum input layouts and the classical controls:

```bash
.qfm-venv/bin/python qfm/controlled_experiment.py --full
```

The five-qubit semantic layout and 15-qubit positional layout use a shared,
explicit parser of this controlled grammar. The positional layout puts each
clue at its word position; non-cue words and padding receive zero angles.
The semantic layout uses five named slots. Neither receives labels, referents,
tiers, scenario IDs, or symbolic metadata as input. The factor metadata is used
only in tests to verify that the rewritten text retains the intended clues.

This is a synthetic representation benchmark. Its semantic combinations all
occur in training, and the parser explicitly recognizes the paraphrase vocabulary.
Its scores do not measure general language understanding or demonstrate a quantum
advantage. See `qfm/README.md` for the full experimental protocol and results.
