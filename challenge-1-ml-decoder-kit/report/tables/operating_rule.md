**Which method at which noise level: an operating rule chosen after the test results were seen, from the measured rows; not a frozen decoder.** X memory only: failures of the 10,000 X test shots at each p, and the mean time per X shot. Our pipeline's time is the X half of its timed rows' own per-shot times on the test shots; the predecoder's is its clean one-thread sample on 1,000 validation X shots (exploration shots); each row says which. Unpaired, so a difference of a few shots is within the noise.

| p | Method | X test failures | Mean time per X shot, ms | Where the time comes from |
|---|---|---|---|---|
| 0.003 | our pipeline (the hierarchy with the network), one core | 1 of 10,000 | 23.7 | test shots, its timed row |
| 0.003 | our pipeline, its two fast decodes on two cores | 1 of 10,000 | 15.0 | test shots, its timed row |
| 0.003 | the predecoder (network and head, then relay-BP with the head's priors) | 2 of 10,000 | 52.9 | validation sample (exploration shots) |
| 0.003 | the predecoder with the fallback (TCBP-JOBS where its relay-BP did not converge; needs a GPU; the rule was designed after the first test result: post hoc at p = 0.005 and 0.004) | 1 of 10,000 | 52.9 | validation sample, plus TCBP-JOBS's GPU decode on the 0.1% of shots that fall back |
| 0.003 | TCBP-JOBS alone (NVIDIA, GPU), for reference | 1 of 10,000 | 2.0 | test shots; a GPU decode call alone |
| 0.003 | TCBP-JOBS with the head's priors (the TCBP version; designed after the test results were seen; no validation at these p), a reference row, not one of the rule's options | 0 of 10,000 | not timed |  |
| 0.004 | our pipeline (the hierarchy with the network), one core | 18 of 10,000 | 104 | test shots, its timed row |
| 0.004 | our pipeline, its two fast decodes on two cores | 18 of 10,000 | 71.7 | test shots, its timed row |
| 0.004 | the predecoder (network and head, then relay-BP with the head's priors) | 15 of 10,000 | 94.0 | validation sample (exploration shots) |
| 0.004 | the predecoder with the fallback (TCBP-JOBS where its relay-BP did not converge; needs a GPU; the rule was designed after the first test result: post hoc at p = 0.005 and 0.004) | 10 of 10,000 | 94.0 | validation sample, plus TCBP-JOBS's GPU decode on the 0.3% of shots that fall back |
| 0.004 | TCBP-JOBS alone (NVIDIA, GPU), for reference | 20 of 10,000 | 3.9 | test shots; a GPU decode call alone |
| 0.004 | TCBP-JOBS with the head's priors (the TCBP version; designed after the test results were seen; no validation at these p), a reference row, not one of the rule's options | 4 of 10,000 | not timed |  |
| 0.005 | our pipeline (the hierarchy with the network), one core | 245 of 10,000 | 420 | test shots, its timed row |
| 0.005 | our pipeline, its two fast decodes on two cores | 245 of 10,000 | 310 | test shots, its timed row |
| 0.005 | the predecoder (network and head, then relay-BP with the head's priors) | 151 of 10,000 | 212 | validation sample (exploration shots) |
| 0.005 | the predecoder with the fallback (TCBP-JOBS where its relay-BP did not converge; needs a GPU; the rule was designed after the first test result: post hoc at p = 0.005 and 0.004) | 107 of 10,000 | 212 | validation sample, plus TCBP-JOBS's GPU decode on the 0.8% of shots that fall back |
| 0.005 | TCBP-JOBS alone (NVIDIA, GPU), for reference | 186 of 10,000 | 8.7 | test shots; a GPU decode call alone |
| 0.005 | TCBP-JOBS with the head's priors (the TCBP version; designed after the test results were seen), a reference row, not one of the rule's options | 72 of 10,000 | 55.8 | an A100 slice, not a clean timing (network on validation shots, build and decode on the test shots); after the speed work about 38 ms (validation shots, each part timed alone; about 52 for the registered path measured the same way) |
