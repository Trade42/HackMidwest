# PriorTilt against relay-BP on the kit's shots (laptop CPU runs)

These are exploration numbers on train and validation shots, not final results. No `test_*` or `go_*` file was opened:
`decode_file.py` refuses them by name.

## What was compared

- **relay-BP**: the kit's `decoders.relay_decoder(H, priors)` with the kit's settings unchanged (`RELAY_SETTINGS`,
  stop after 5 solutions), from `relay-bp==0.2.2`.
- **PriorTilt**: the Rust core of PriorTilt (Ye and Delfosse, IonQ, arXiv:2611.13373) with the paper's three-phase
  schedule for BB144 at the file's p (`priortilt.params.tuned_config("BB144", p)`): max_rounds 3000, initial_iters 60,
  round_len 6, f64 messages.
  - Code: https://github.com/ionq-publications/PriorTiltDecoder at commit `8eefd37d694b5942fe60ec6d5c82ce97f3d2a450`.
  - Licence: CC BY-NC-SA 4.0. **None of its code is included here.** It was used as a tool, built locally with
    `uv pip install ./decoder` (maturin).
  - The seed scheme is per 500-shot chunk: P1 = seed 1 + chunk index, P2 = seed 1001 + chunk index.
- Both decode **the kit's own problem**:
  - H (936 x 8,784), L (12 x 8,784) and priors come from `circuit.dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis)`.
  - The syndrome is `detectors[:, basis_detectors]` of each shot file.
  - PriorTilt's core is fed this H and these priors directly (`priortilt._create_core`), not its own circuits.
- **Shot order**: each file is cut into chunks of 500 shots, decoded in file order, each chunk on a fresh decoder.
- **Scoring**: a shot fails when any of the 12 predicted observables (L e mod 2) is wrong; an unconverged shot is scored
  on what the decoder returned.
- **Timing**: only the decode call is timed, as in the kit's `decoders.score()`.

PriorTilt's own BB144 circuits (`StimCircuit/`) are a close neighbour of the kit's problem, not the same one.
`compare_problem.log` and `compare_noise_X_p0.004.log` show the comparison:

- The same:
  - gates: 84 CX layers, 97 ticks;
  - H shape and nnz (30,672);
  - column-weight and row-weight histograms.
- Different:
  - noise placement: the kit has X_ERROR/Z_ERROR after resets and noiseless M/MX, while PriorTilt has M(p)/MX(p) and
    more DEPOLARIZE1;
  - as a result, 936 of the 8,784 priors differ.
- Expected faults per shot nearly match: 47.31 for the kit against 46.95 for PriorTilt's circuit at p = 0.004.

## Machine and timing caveat

- Hardware: ASUS ROG Zephyrus G14 (GA403UV), AMD Ryzen 9 8945HS (8 cores, 16 threads), 15.3 GiB RAM.
- OS: Windows 10 Home 10.0.19045.
- Software: Python 3.12.13; numpy 2.5.5, scipy 1.18.1, stim 1.16.0, ldpc 2.4.1, relay-bp 0.2.2 (built from source),
  priortilt_rs 1.0.0.
- Build tools: Rust 1.91.0 and MSVC Build Tools 2022.

**Every time here was taken beside a GPU training run on the same laptop.**

- 1 to 5 decode processes ran side by side, at BelowNormal priority on logical CPUs 0-11.
- The runner's `runner_*.jsonl` records the number of processes running at each job's start and the most during it.
- Compare times with each other, not as absolute latencies.

## Files

| File | What it is |
| --- | --- |
| `decode_file.py` | Decodes a range of 500-shot chunks of one shot file with relay-BP or PriorTilt. Saves per shot: failed, seconds, iterations, converged, the 12 predicted bits, the truth, the number of faults, and the weight sum log((1-p)/p) |
| `runner.py` | Runs `decode_file.py` jobs, at most 6 side by side. Guards: no new job under 1.5 GiB available RAM or at GPU >= 85 C; pause and requeue under 512 MiB. Runs below normal priority |
| `analyze.py` | Paired relay-BP against PriorTilt tables: failures, 2 x 2 table, McNemar exact test, Wilson intervals, time and iterations |
| `analyze_8b.py` | Step 8b: merges R, P1, P2 per file into `results_8b/selector_<file>.npz` and scores the selection rules |
| `analyze_network_select.py` | Joins the 8b files with the cascade network's logits (`../cascade_X_s1/eval/`) and scores network-based selection rules |
| `compare_problem.py`, `compare_noise.py` (+ `.log`) | PriorTilt's BB144 circuits against the kit's problem |
| `make_step9_data.ps1` | Generates the step-9 shot files with the kit's `make_dataset.py` and writes the step-9 job list |
| `results_p0.004_X.txt`, `results_p0.004_Z.txt` | Step 8: relay-BP against PriorTilt (P1) at p = 0.004, per file and per basis |
| `step6_validation_X.txt` | Step 6 sanity check: relay-BP fails 31 of 5,000 (0.62%) on validation_X |
| `results_8b.txt` | Step 8b: R, P1, P2 and the selection rules, for X, Z and pooled |
| `results_network_select_step30000.txt` | The network at step 30,000 as a selector, X memory, 25,000 shots |
| `results_8b/selector_{validation_X,train_X,validation_Z,train_Z}.npz` | Per shot, in file order: `truth` [n,12]; for each of R, P1, P2: `predicted` [n,12], `converged`, `iterations`, `n_faults`, `weight`, `seconds`, `failed`; `settings_*`; `data_sha256` |
| `jobs_*.txt`, `runner_*.jsonl` | Job lists and per-job records (start, end, exit code, side-by-side counts) |
| `data_sha256_p0.004.txt`, `data_sha256_step9.txt` | sha256 of every input file |

The scripts use absolute paths of the laptop (`C:\Users\salav\Salavat\hackMidwest\...`). The checkouts they expect:

- `repo-main` (branch main at c54ce0d), for the kit;
- `external\PriorTiltDecoder`;
- the venv `.venv-cpu`.

## Commands

From `C:\Users\salav\Salavat\hackMidwest`:

```
# step 5: does PriorTilt's circuit match the kit's problem
.venv-cpu\Scripts\python.exe -I compare_problem.py <kit> <PriorTiltDecoder>
.venv-cpu\Scripts\python.exe -I compare_noise.py <kit> <PriorTiltDecoder> X 0.004

# steps 6 and 8 (relay-BP and P1 on the four p = 0.004 files), 8b (R, P1, P2), 9 (p = 0.005 and 0.003)
.venv-cpu\Scripts\python.exe -I -B runner.py --name p0.004 --workers 5 --jobs jobs_p0.004.txt
.venv-cpu\Scripts\python.exe -I -B runner.py --name 8b --workers 5 --results results_8b --jobs jobs_8b.txt
.venv-cpu\Scripts\python.exe -I -B runner.py --name step9 --workers 5 --results results_step9 --jobs jobs_step9.txt

# tables
.venv-cpu\Scripts\python.exe -I -B analyze.py "p=0.004 X" validation_X train_X
.venv-cpu\Scripts\python.exe -I -B analyze.py "p=0.004 Z" validation_Z train_Z
.venv-cpu\Scripts\python.exe -I -B analyze_8b.py results_8b validation_X train_X validation_Z train_Z
.venv-cpu\Scripts\python.exe -I -B analyze_network_select.py 30000
```

A runner job runs `decode_file.py --kit <kit> --pt <PriorTiltDecoder> --data <file> --decoder relay|priortilt
[--seed-base 1001] --chunks k:k+1 --out <result>`.

Reproducibility check: R and P1 of step 8b equal step 8 shot by shot, in both failures and iterations.

## Input files and their sha256

From repo-main `challenge-1-ml-decoder-kit/data/p0.004/` (branch main at c54ce0d; not copied here):

- `train_X.npz`: 20,000 shots, sha256 98c0afeb2a0b6b27f3b689e832160a060a4c0aeb37d229f84daa1995a9a0f7cc
- `train_Z.npz`: 20,000 shots, sha256 14dce11586fa51256f476224d0fe3965e0562f2d8f68cba6192883a65fb6b24a
- `validation_X.npz`: 5,000 shots, sha256 b00ee09c35ed088a12bcba60b8b704b528c47fb156c7d09cf0f171f90a181f3d
- `validation_Z.npz`: 5,000 shots, sha256 843c4daf24e2b56c5988670550bbcdbac8f50b18dbb9d24beb64969146cb8e20

Generated on the laptop with the kit's `make_dataset.py` (repo-main, unchanged), command
`python make_dataset.py --basis <b> --p <p> --shots <n> --seed <s> --out <file>`.

**These exist only on the laptop**, in `C:\Users\salav\Salavat\hackMidwest\runs\priortilt\data\`. They are not in this
branch:

| File | p | Basis | Seed | Shots | Bytes | sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| `p0.005\gen_X_seed9205.npz` | 0.005 | X | 9205 | 20,000 | 4,219,522 | 45a9761e688088c0e502e80a5569258985242e40231eea7f474f7887188213eb |
| `p0.005\gen_Z_seed9206.npz` | 0.005 | Z | 9206 | 20,000 | 4,211,160 | 4069ac5fbd20def095dad050b4f94e9929eb505af068f636cbb87a07ca87d147 |
| `p0.003\gen_X_seed9203.npz` | 0.003 | X | 9203 | 50,000 | 8,337,014 | 4625e704682b3a6ef5c465eec33479d04f3d6f9a62912d4e4ce176e51bf7725e |
| `p0.003\gen_Z_seed9204.npz` | 0.003 | Z | 9204 | 50,000 | 8,333,938 | b7230c3bec89bbbbd7df8bf18ce674657b4f99190a53c3de9f04928d85859af1 |

Stim seeds give different shots on different machines, so these files cannot be regenerated byte for byte elsewhere.
