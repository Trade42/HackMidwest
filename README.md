# Quantum-assisted document translation

The `translation/` demo uses MarianMT to propose English-to-Spanish translations for each sentence. It scores candidates by average token log probability and uses Marian cross-attention as a rough alignment for English terms repeated across sentences. A QUBO penalizes inconsistent target terms and enforces one candidate per sentence. Exact enumeration, local search, and PennyLane QAOA solve that same QUBO.

```bash
uv venv .translation-venv
uv pip install --python .translation-venv/bin/python -r translation/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.translation-venv/bin/python -m translation.main --text 'The technician inspected the seal on the pump. The seal was damaged.'
```

The first run downloads `Helsinki-NLP/opus-mt-en-es`. Use `--candidates 2`, `--layers 2`, `--steps 40`, and `--shots 1000` to adjust the demo. The printed objective measures the constructed score; it does not establish translation accuracy or quantum advantage. QAOA timing is for a simulator.
