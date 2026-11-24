# Project guide

This guide explains the pieces of the Gross-code decoder project and how to read its results. Start with the [README](README.md) for a short summary, the [main decoder explanation](fpga_board/MAIN_ROUTER.md) for the measured software comparison, or the [FPGA demo](fpga_board/FPGA_DEMO.md) for the board prototype.

## The experiment

The supplied [decoder kit](challenge-1-ml-decoder-kit/README.md) simulates X and Z quantum-memory experiments with the Gross `[[144, 12, 12]]` code. Each shot prepares the memory, runs 12 noisy check cycles, and measures the data. The circuit produces 1,728 detector bits, which mark unexpected changes in check results, and 12 logical-observable bits used to score the answer. The kit baselines decode the 936 check bits for the selected memory basis. A shot fails if any predicted logical bit is wrong. X and Z shots are paired for the reported score; a pair fails if either shot fails.

The required physical error rates are `p = 0.003`, `0.004`, and `0.005`. The project uses separate saved training, validation, and held-out test files for each rate and basis. Paired comparisons use the same saved test shots, with model settings frozen before testing. Logical error rate per cycle is derived from the fraction `P` of failed pairs as `1 − (1 − P)^(1/12)`; it is different from the fraction of individual shots that fail. The [kit README](challenge-1-ml-decoder-kit/README.md) describes the baseline circuits and scoring, and the [reproduction guide](challenge-1-ml-decoder-kit/report/REPRODUCE.md) records file hashes and commands.

## What the implementations do

| Component | Purpose | Evidence and limits |
| --- | --- | --- |
| [Main hierarchy](challenge-1-ml-decoder-kit/hierarchy/README.md) | Runs PriorTilt twice; on disagreement, adds Relay-BP and chooses a candidate with a neural ranking. | Held-out software comparisons are described in the [main decoder explanation](fpga_board/MAIN_ROUTER.md). The neural rule's incremental gain over classical ranking is uncertain. |
| [Neural model](challenge-1-ml-decoder-kit/neural/README.md) | Reads detector bits and scores logical answers for the hierarchy's difficult shots. | Training and evaluation are separate from the timed full-decoder result. |
| [First-solution router](ROUTER_STUDY.md) | Uses a learned risk score to decide when a first Relay-BP solution needs a full rerun. | The backup software study has paired comparisons for two rates; its result is separate from the main hierarchy. |
| [Urbana FPGA router](fpga_board/README.md) | Implements the backup router's 17-feature score and decision in hardware. | Saved-shot RTL simulation and synthesis estimates are available; board operation and end-to-end decoder acceleration have not been demonstrated here. |
| [FPGA neural study](fpga_network/README.md) | Explores fixed-point execution of the neural model. | Arithmetic simulations and an unrouted synthesis estimate; no memory controller or board bitstream. |
| [Atom-loss study](challenge-1-ml-decoder-kit/loss/RESULTS.md) | Simulates flagged atom loss and tests how loss information changes Relay-BP decoding. | Separate exploration shots under a simplified loss model; an oracle that knows the loss time is an upper bound, not an implemented decoder. |
| [ML-first experiment](challenge-1-ml-decoder-kit/ml_decoder/README.md) | Tries a neural prediction before Relay-BP fallback. | Exploratory alternative, not the measured main hierarchy. |

The [analysis results](challenge-1-ml-decoder-kit/analysis/RESULTS.md) record the failure studies and candidate comparisons that informed the hierarchy. The [earlier research notes](RESEARCH_NOTES.md) preserve the November 22 planning snapshot and reading list; their planned work and exploration counts should not be treated as the current held-out result.

## Reading the measurements

- **Held-out software comparison:** The main hierarchy and full Relay-BP were evaluated on the same 10,000 X/Z pairs at each rate. At `p = 0.004`, they failed on 44 and 83 pairs respectively; mean complete decode times were 216.46 and 319.46 ms per pair. See the [full table and paired comparison](fpga_board/MAIN_ROUTER.md#what-was-measured).
- **Backup router:** The earlier first-solution policy had the same 84 failed pairs as full Relay-BP in a Mac paired run at `p = 0.004`, with 5.47% lower mean time. That result does not measure the FPGA. See the [router study](ROUTER_STUDY.md#evaluation-design-and-interpretation).
- **FPGA checks:** The backup router's saved-shot decisions matched the Python rule in local simulation. Yosys resource counts are synthesis estimates; they are not Vivado routed utilization or clock timing. See the [board README](fpga_board/README.md#verify-locally).
- **Loss results:** The loss experiments use Clifft and a stated per-CX loss model. The reported X-memory failures are per shot, not X/Z pairs. See the [loss tables](challenge-1-ml-decoder-kit/loss/RESULTS.md).

## Repository map

| Path | Contents |
| --- | --- |
| [`challenge-1-ml-decoder-kit/`](challenge-1-ml-decoder-kit/README.md) | Supplied experiment, saved shots, baselines, evaluators, and decoder implementations |
| [`challenge-1-ml-decoder-kit/report/`](challenge-1-ml-decoder-kit/report/README.md) | Report sources, data provenance, tables, figures, and reproduction steps |
| [`fpga_board/`](fpga_board/README.md) | Trained backup router, RTL, vectors, and Vivado scripts for Urbana |
| [`fpga_router/`](fpga_router/README.txt) | Earlier 16-feature hardware router and its verification record |
| [`fpga_network/`](fpga_network/README.md) | Fixed-point neural-network arithmetic study |
| [`server-jobs/`](server-jobs/README.md) | Scripts used by the team's remote job runner |
