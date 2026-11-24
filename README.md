<img width="2172" height="724" alt="Qubits & Gradients project banner" src="assets/qubits-and-gradients.png" />

# Gross-code quantum error correction

This repository contains a team project for [Infleqtion's MidQWest 2026 Challenge 1](CHALLENGE_BRIEF.md): improve decoding of the `[[144, 12, 12]]` Gross quantum error-correcting code with machine learning. A decoder reads warning signs from repeated quantum checks and predicts which of 12 encoded information bits were flipped. The project compares software decoders, studies atom loss, and tests whether parts of the decision process fit on an FPGA.

## The decoder

The main software decoder runs the published PriorTilt decoder twice with different random seeds. If both answers agree, it returns that answer. If they disagree, it runs Relay-BP for a third candidate and ranks the candidates with a neural network trained on simulated shots. The network chooses among candidate answers; it does not replace the underlying decoders. A separate, smaller logistic model routes Relay-BP's first answer to either acceptance or a full rerun. That backup router is the one implemented in the Urbana FPGA demo.

## Measured software results

On the same held-out files, each row below decodes 10,000 X-memory shots and 10,000 Z-memory shots, paired by position. A pair fails if either answer is wrong. The times include the complete sequential software decode of both shots on the ICRN server; training and startup are excluded. Physical error rate `p` is the simulated noise setting.

| `p` | Decoder | Failed pairs / 10,000 | Mean time per pair |
| --- | --- | ---: | ---: |
| 0.003 | Full Relay-BP | 5 | 85.33 ms |
| 0.003 | Main hierarchy | 3 | 45.74 ms |
| 0.004 | Full Relay-BP | 83 | 319.46 ms |
| 0.004 | Main hierarchy | 44 | 216.46 ms |
| 0.005 | Full Relay-BP | 845 | 1,470.86 ms |
| 0.005 | Main hierarchy | 479 | 824.58 ms |

At `p = 0.004`, the full hierarchy had 39 fewer failed pairs and a 32.24% lower mean time than full Relay-BP. Only five baseline pairs failed at `p = 0.003`, so the difference of two pairs there is weak evidence about accuracy. The neural ranking's improvement over a classical candidate-ranking rule was not conclusive in the paired comparisons. At `p = 0.005`, the hierarchy's mean time improved while its 99th-percentile time worsened. [Method, full table, and attribution](fpga_board/MAIN_ROUTER.md#what-was-measured)

These are **software** results. The [Urbana FPGA demo](fpga_board/FPGA_DEMO.md) is designed to replay saved X-memory shots through the backup router and check its decisions against the Python model. Local simulation matched all 5,000 saved validation shots; no physical-board result is documented. The [neural-network FPGA study](fpga_network/README.md) reports simulation and synthesis estimates, without a board bitstream or routed timing. [Atom-loss experiments](challenge-1-ml-decoder-kit/loss/RESULTS.md) use a separate, simplified simulated loss model and should not be read as measurements on Infleqtion hardware.

## Explore the repository

| Start here | What it covers |
| --- | --- |
| [Project guide](information.md) | The experiment, components, result boundaries, and directory map |
| [Main decoder explanation](fpga_board/MAIN_ROUTER.md) | How the hierarchy works, measured results, and what each component contributes |
| [FPGA demo guide](fpga_board/README.md) | The backup router's hardware interface, local checks, and Urbana setup |
| [Decoder kit](challenge-1-ml-decoder-kit/README.md) | Baseline circuits, setup, scoring, and sample commands |
| [Reproduction guide](challenge-1-ml-decoder-kit/report/REPRODUCE.md) | How the saved-shot comparison and report artifacts were produced |
| [Original challenge brief](CHALLENGE_BRIEF.md) | Both MidQWest challenge descriptions and requirements |
| [First-solution router study](ROUTER_STUDY.md) | Earlier paired analysis of the backup software router |

To try the supplied baseline experiment, use Python 3.12 and run the commands in the [decoder kit setup](challenge-1-ml-decoder-kit/README.md#setup). The saved-shot evaluation and its extra dependencies are documented in the [evaluation guide](challenge-1-ml-decoder-kit/evaluation/README.md).
