# tcbp-fastpath: the frozen rule

Exploration, after the test results; a separate branch (`tcbp-fastpath`). Chosen on validation_X at p = 0.005 only
(5,000 Stim shots, seed 201). No test file was opened. The commit times decide.

## The rule now frozen: C9, the head's own entropy (`CRITERION_4.md`, commit `7bcd7058`)

For each X shot at p = 0.005:
1. Run the network and head.
2. Compute s = the sum over the 8,784 columns of the binary entropy, in bits, of sigmoid(logit). The logits are cast
   to float16 first.
3. If s <= **69.14651567152464** bits, answer with TCBP-JOBS with the kit's priors, one decoder built once.
4. Otherwise answer with the TCBP version: TCBP-JOBS built with the shot's head priors.

On validation (`result3_p0.005.json`): 35.4% of the shots take the fast path, with 0 added failures (TH's 33, shot
for shot). The mean is 38.8 ms against 51.1 ms for TH alone, and the p99 is the same 63.1 ms (round 2's 9.4 ms
network). **The held-out check** (fresh Stim shots, seed 2026112251, `CRITERION_4.md`; `result_holdout_p0.005.json`):
holds with a cost. 48 failures against TH's 48 (1 added, 1 removed, p = 1.0); 35.3% of shots on the fast path;
33.5 ms mean against 46.2 ms for TH alone, p99 68.2 against 67.6 (today's component times, network 1.7 ms).

**For p = 0.005 only.** Applied unchanged at p = 0.004 (`README.md`, "p = 0.004 validation"), 69.1 bits sends 87.5% of
the shots to F and adds 2 failures. A deployment needs a threshold per p, or one scaled with p. At p = 0.004 the
criterion chose C2 (syndrome weight <= 127, 64% fast, 24.2 ms mean). That choice rests on 12 F-only shots and is not
frozen.

## How it was chosen (three looks at the same validation shots)

| round | criterion (commit) | chosen | mean ms | p99 ms | note |
|---|---|---|---|---|---|
| 1 | `CRITERION.md` (`b0a410e6`) | C2 syndrome weight <= 115 | 49.3 | 63.1 | under the committed time model 61.9 / 102.9; corrected (disclosed in the README) |
| 2 | `CRITERION_2.md` (`17f86351`) | C6 F's decode time <= 5.51 ms | 43.4 | 102.9 | a threshold in ms that does not carry between runs |
| 3 | `CRITERION_3.md` (`cbfbfbcd`) | **C9 the head's entropy <= 69.1 bits** | **38.8** | **63.1** | frozen in `CRITERION_4.md` |

Each round's choice is by the same declared rule: 0 added failures against TH on all 5,000 shots, then the lowest
mean. Every threshold sits on the edge of these shots. C9's half split adds 1 failure one way and 0 the other.

## Recommendation (kept apart from the choice)

On the held-out shots the saving held: 28% of the mean time, at the same accuracy and p99. If it goes into the
pipeline, it should go in for X at p = 0.005 only. Before that, F's time should be measured on a decoder built once
and running alone. A Z rule, or one for another p, needs its own threshold fitted on that p's validation shots.
