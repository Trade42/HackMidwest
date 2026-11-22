# Independent review of the Iceberg encoder

These are the raw scripts and logs of a separate reviewer agent. It was given the encoder
and the claims in the main README, not the tests or the build history, and it wrote its
own simulator (`vlib.py`) and its own Pauli tracker (`faultlib.py`, `tracker_large*.py`).

They are kept as evidence, not as part of the package. `pytest` does not collect them
(`testpaths = ["tests"]`), and CI does not run them.

What they found, in order:

1. Claims 1 to 4 confirmed (`test_claim1.py` to `test_claim5.py`, `out_claim*.txt`).
2. Gap 1: with two or more layers, one Y fault on a bus qubit passed every check
   (`repro_bus_hole.py`, `repro_bus_hole_qaoa.py`, `repro_min.py`). Fixed by the two-step
   witness in `src/iceberg/bus.py`.
3. Re-run after the fix (`rerun_*.py`, `out_rerun_*.txt`, `out_tracker_large*.txt`):
   clean for 3 or more paragraphs.
4. Gap 2: one or two paragraphs still failed, because the bus row was too short
   (`repro_k2.py`). Fixed by padding to four code-logical qubits.

The `out_*.txt` logs are from the code as it was when each script ran, so the early ones
show the gaps. On the current code:

- `repro_min.py` and `repro_bus_hole.py` report a kept share of 0: the fault is discarded.
- `repro_k2.py` stops with `StopIteration`. It hard-codes qubit numbers of the old
  two-qubit bus row, which no longer exists.

The regression tests for both gaps live in `tests/test_iceberg_faults.py`.

To run a script: `pip install -e ".[dev]" stim`, then `cd review && python <script>.py`.
