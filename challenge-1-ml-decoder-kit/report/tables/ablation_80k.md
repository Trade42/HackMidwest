**The ablation at scale, on exploration shots never used for a fit**: 80,000 fresh X-memory shots at p = 0.004 (Stim seeds 9401 to 9416), decoded in session mode as the plugin decodes, the final X network (step 120,000). One shot is one X-memory shot, not a pair. The two PriorTilt runs differ on 366 shots; on 302 of them at least one candidate is right, and the last column counts how often each tie-break picks a right one there.

| Decoder | Whose work | Failures | Share of shots (95% interval) | Right pick where one existed |
|---|---|---|---|---|
| PriorTilt, one run (seed A) | IonQ | 357 of 80,000 | 0.446% (0.402% to 0.495%) |  |
| PriorTilt, one run (seed B) | IonQ | 348 of 80,000 | 0.435% (0.392% to 0.483%) |  |
| our hierarchy, lowest-weight tie-break | ours, nothing learned | 298 of 80,000 | 0.372% (0.333% to 0.417%) | 191 of 302 |
| our hierarchy, lowest weight among the converged candidates | ours, nothing learned | 253 of 80,000 | 0.316% (0.280% to 0.358%) | 236 of 302 |
| our hierarchy, neural tie-break (final X network) | ours, learned | 237 of 80,000 | 0.296% (0.261% to 0.336%) | 252 of 302 |
| ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | needs the truth | 187 of 80,000 | 0.234% (0.203% to 0.270%) |  |

Paired, on the same shots:

| Comparison | Both fail | Only the first fails | Only the second fails | Exact p |
|---|---|---|---|---|
| the neural tie-break against lowest weight | 211 | 26 | 87 | 7e-09 |
| the neural tie-break against lowest weight among the converged | 198 | 39 | 55 | 0.12 |
| lowest weight among the converged against lowest weight | 252 | 1 | 46 | 7e-13 |

Source: `report/data/exploration/ablation_80k/per_shot.npz` (recomputed here) and `summary.json` (checked against it), from branch `hierarchy` at commit `534d428`. Relay-BP alone was not decoded on these shots.
