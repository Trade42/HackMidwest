# Data

Three disjoint question files, all in the HotpotQA distractor setting (10 paragraphs
per question, 2 of them hold the facts needed for the answer).

| File | Questions | Seed | Bridge | Comparison | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| `questions.jsonl` | 150 | 0 | 115 | 35 | evaluation, every lane and tier |
| `calibration.jsonl` | 50 | 1 | 42 | 8 | tuning `tau` and `theta`, never scored |
| `demo.jsonl` | 3 | 2 | 3 | 0 | live demo (short bridge questions) |

One JSON object per line:

```json
{"qid": "...", "question": "...", "answer": "...", "type": "bridge", "level": "hard",
 "paragraphs": [{"title": "...", "text": "..."}], "gold": [5, 9]}
```

`gold` holds the indices of the two supporting paragraphs.

## Where the questions come from

The files here were drawn from the 1,000-question sample of the HotpotQA dev
distractor set that ships with the HippoRAG repository:

    https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset/hotpotqa.json

That sample keeps the original HotpotQA record format. All 1,000 questions are
level "hard" (811 bridge, 189 comparison) and 991 of them have exactly 10
paragraphs; only those 991 are eligible. It was used because the official
download hosts were not reachable from the machine that built this repository.

HotpotQA is distributed under CC BY-SA 4.0 (Yang et al., "HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering", EMNLP 2018). These files are
a redistribution of a subset under the same licence.

## Regenerating

Same files as committed (needs the HippoRAG sample saved locally):

    python scripts/prepare_data.py --source path/to/hotpotqa.json

The official dev distractor set (7,405 questions) can be used instead. With no
`--source`, the script fetches it through `datasets`, then the Hugging Face hub,
then the HotpotQA server:

    python scripts/prepare_data.py

Sampling keeps records with exactly 10 paragraphs and 2 gold paragraphs, sorts
them by id, and draws with `numpy.random.default_rng(seed)`, so the result does
not depend on the order of the source file. A different source gives a different
set of questions.

## Scores

`scores/{qid}.npz` holds, for every question in the three files, the relevance
`r` (10 values) and the pairwise overlap `s` (10 x 10) from the scorer named in
the `scorer` field. `reproduce.py` writes missing score files with the scorer in
the experiment config. The committed scores come from the offline scorer of
`configs/offline.yaml`: TF-IDF relevance with a second-hop term (`hop: 3`) and
question-coverage overlap (`overlap: coverage`). Plain TF-IDF puts both gold
paragraphs in its top 3 for 56% of the calibration questions; with the second-hop
term it is 66%.
