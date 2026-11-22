# Training runs (laptop GPU)

Common to every run below unless stated:

- Machine: Windows 10 Home 10.0.19045, NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, sm_89), driver 621.24, WDDM.
- Python 3.12.13; torch 2.14.1+cu130 (CUDA 13.0, cuDNN 92400); stim 1.16.0; numpy 2.5.5; scipy 1.18.1.
- Run from `challenge-1-ml-decoder-kit/` with `PYTHONDONTWRITEBYTECODE=1` (the kit tracks `__pycache__/*.pyc`).
- Seed: none. `train_logical_cnn.py` sets no torch or numpy seed, so no run is bit-for-bit repeatable.
- Metrics are on DEM-sampled shots (20,000 fresh held-out shots per p, drawn at report time), not on kit Stim shots.
- GPU memory is nvidia-smi `memory.used` for the whole GPU, sampled every 2 s (the trainer does not report it).
- Logs live outside the repo, in `C:\Users\salav\Salavat\hackMidwest\runs\`.

## x_check3min: 3-minute check, X basis

- Commit: 0eab7dd90b3152c35506542e1e586587e30b7956 (clean).
- Command: `python ml_decoder/train_logical_cnn.py --basis X --minutes 3 --width 32 --depth 2 --batch 256`
- Config: width 32, depth 2, kernel 5, batch 256, lr 3e-4, curriculum over p in {0.0005, 0.001, 0.002, 0.003, 0.004, 0.005}. 0.7 M parameters.
- Started 2026-11-21 18:27:23 CST. Wall time 193.8 s (about 12 s of DEM setup, 180 s of training). Exit 0, stderr empty.
- Throughput: 5.8 M training shots in 180 s, about 32,000 shots/s.
- GPU memory peak: 211 MiB (baseline 0 MiB). GPU about 51% busy on average; peak 69 C.
- Final report:

| p | Shots wrong | Bits right | Always-0 guess, bits right |
| --- | --- | --- | --- |
| 0.003 | 99.60% | 54.72% | 50.51% |
| 0.004 | 99.92% | 50.98% | 50.25% |
| 0.005 | 99.96% | 50.12% | 49.87% |

- Loss (running average of BCE): 0.6841.
- Model: `runs\x_check3min.model_X.pt` (moved out of the kit so the 4-hour run starts fresh).

### Low-p eval of the check model (2026-11-21 19:23 CST)

- Copy: `runs\x_check3min.evalcopy.pt` (same sha256 as the original). Command as for the 0.83 h checkpoint below, `--p-report 0.0005,0.001,0.002,0.003 --shots 20000`. Wall 12.1 s, exit 0.

| p | Shots wrong | Bits right | Always-0 guess, bits right |
| --- | --- | --- | --- |
| 0.0005 | 23.90% | 96.26% | 73.20% |
| 0.001 | 62.84% | 86.17% | 60.71% |
| 0.002 | 95.20% | 65.71% | 52.34% |
| 0.003 | 99.65% | 54.78% | 50.72% |

## x_4h: 4-hour X run

- Commit: 0eab7dd90b3152c35506542e1e586587e30b7956 (clean).
- Command: `python ml_decoder/train_logical_cnn.py --basis X --hours 4 --out model_X.pt`
- Config: width 128, depth 6, kernel 5, batch 2048, lr 3e-4, same curriculum. 5.6 M parameters.
- Started 2026-11-21 18:31:14 CST (00:31:14 UTC). At 25 s: 4077 MiB used, 100% busy.
- **Stopped** at 19:26:48 CST on the orchestrator's decision, after 3,334.6 s wall (exit -1, killed; stderr empty). The model stayed at the always-0 guess at every p (see the low-p eval below), and from about 77 minutes the schedule trains only at p >= 0.003.
- GPU memory peak: 4966 MiB, of which about 4079 MiB was training; the rest came from two evals that shared the GPU.
- Kept as the record of a failed configuration: `runs\x_4h.log`, `x_4h.gpu.csv`, `x_4h.summary.txt`, and `x_4h.ckpt_0.83h.pt` (22,249,247 bytes, sha256 6958513cda0b382b1d7d6a4bebe2dbaa8e863624043bac6043cd110d59c76b27).
- Checkpoint: `challenge-1-ml-decoder-kit/model_X.pt`, saved every 10 minutes.
- Throughput: about 2,500 shots/s. The GPU is 99-100% busy at 4079 MiB and 76-78 C, SM clock 2610 of 3105 MHz, with no thermal or power-cap slowdown.

| Time | M shots | Loss | p=0.003 bits right (always-0) | p=0.004 | p=0.005 | Shots wrong at 0.003 / 0.004 / 0.005 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.17 h | 1.5 | 0.6376 | 50.38% (50.38%) | 50.27% (50.27%) | 49.77% (49.77%) | 99.97% / 99.98% / 99.98% |
| 0.33 h | 3.0 | 0.6518 | 50.38% (50.57%) | 50.18% (50.07%) | 50.05% (49.97%) | 99.98% / 99.97% / 99.99% |
| 0.50 h | 4.4 | 0.6419 | 50.41% (50.63%) | 50.16% (50.24%) | 50.00% (50.12%) | 99.98% / 99.95% / 99.97% |
| 0.67 h | 5.9 | 0.6775 | 50.82% (50.58%) | 50.11% (50.08%) | 50.01% (50.02%) | 99.95% / 99.97% / 99.97% |
| 0.83 h | 7.4 | 0.6829 | 50.67% (50.66%) | 49.98% (49.98%) | 50.23% (50.23%) | 99.94% / 99.98% / 99.98% |

### Low-p eval of the 0.83 h checkpoint

- Copy: `runs\x_4h.ckpt_0.83h.pt` (22,249,247 bytes, same sha256 as model_X.pt written at 19:21:29 CST).
- Command: `python ml_decoder/train_logical_cnn.py --basis X --eval <copy> --p-report 0.0005,0.001,0.002,0.003 --shots 20000`, run alongside training. Wall 34.7 s, exit 0.

| p | Shots wrong | Bits right | Always-0 guess, bits right |
| --- | --- | --- | --- |
| 0.0005 | 86.09% | 72.85% | 72.86% |
| 0.001 | 97.63% | 60.78% | 60.76% |
| 0.002 | 99.87% | 52.30% | 52.31% |
| 0.003 | 99.98% | 50.64% | 50.64% |

- Reading: the model predicts 0 for every shot, even at the lowest p it trained on first. By then it had taken about 3,600 optimizer steps (batch 2048), against about 22,600 in the 3-minute check (batch 256).

## x_1h_w32: 1-hour X run, check configuration

- Commit: 0eab7dd90b3152c35506542e1e586587e30b7956 (clean).
- Command: `python ml_decoder/train_logical_cnn.py --basis X --hours 1 --width 32 --depth 2 --batch 256 --p-report 0.0005,0.001,0.002,0.003,0.004,0.005 --out C:\Users\salav\Salavat\hackMidwest\runs\x_1h_w32.model_X.pt`
- Config: width 32, depth 2, kernel 5, batch 256, lr 3e-4, default clock curriculum (top window {0.003, 0.004, 0.005} from about 19 minutes). 0.7 M parameters.
- Chosen because the 3-minute model was well above the guess at low p (+23.1 points at p = 0.0005, +25.5 at p = 0.001).
- Started 2026-11-21 19:27:11 CST (01:27:11 UTC). At 25 s: 211 MiB used, 54% busy.
- Pass line, fixed before launch (orchestrator), judged at p = 0.003 on the final line:
  - PASS: shots wrong is 90% or less.
  - KILL: bits right is under 60%, or it moved less than 1 point over the last 30 minutes.
  - Otherwise the orchestrator decides on a mid-size model.

| Time | M shots | Loss | Bits right (always-0) at p=0.0005 | 0.001 | 0.002 | 0.003 | 0.004 | 0.005 | Shots wrong at 0.003 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.17 h | 19.6 | 0.3534 | 99.76% (72.96%) | 97.41% (60.90%) | 79.76% (52.38%) | 59.90% (50.53%) | 51.99% (50.26%) | 50.30% (49.90%) | 96.82% |
| 0.33 h | 42.1 | 0.6551 | 99.74% (73.21%) | 97.82% (60.79%) | 83.89% (52.24%) | 64.53% (50.47%) | 53.52% (50.22%) | 50.64% (50.20%) | 93.14% |
| 0.50 h | 64.2 | 0.6406 | 99.59% (73.37%) | 97.89% (60.82%) | 84.44% (52.46%) | 65.42% (50.28%) | 54.35% (50.17%) | 50.74% (50.17%) | 92.98% |
| 0.67 h | 84.3 | 0.6370 | 99.69% (73.13%) | 98.08% (60.81%) | 85.93% (52.42%) | 66.71% (50.62%) | 54.91% (50.10%) | 51.23% (49.95%) | 91.52% |
| 0.83 h | 103.0 | 0.6317 | 99.74% (73.20%) | 98.28% (60.87%) | 86.23% (52.49%) | 67.42% (50.67%) | 55.31% (49.97%) | 50.95% (50.09%) | 90.71% |
| done (1.00 h) | 117.3 | 0.6420 | 99.74% (73.25%) | 98.34% (60.77%) | 86.49% (52.44%) | 67.76% (50.54%) | 55.75% (50.07%) | 51.13% (49.96%) | 90.55% |

- Ended 20:27:26 CST. Wall 3,615.0 s, exit 0, stderr empty. About 32,600 shots/s on average. In the last 10 minutes it dropped to about 23,800 shots/s while 5 CPU decode processes (the PriorTilt study) shared the CPU.
- GPU memory peak 272 MiB; GPU about 51% busy; max 74.3 C.
- Checkpoint: `runs\x_1h_w32.model_X.pt`, 2,848,081 bytes, sha256 26fcde3ec30ec8ba0ac5ccc02b30ce27614216453cbc83c076c4ddcba5db3b46.
- Verdict at p = 0.003: 90.55% of shots wrong (not PASS: needs 90% or less). 67.76% of bits right, +2.34 points over the last 30 minutes (not KILL). So it is between the lines.

## x_3h_w32_resume (run 3B): w32 model continued for 3 hours at p = 0.003-0.005

- Commit: 0eab7dd (clean). Resumes from `runs\x_1h_w32.resumecopy.pt` (same sha256 as the run-2 checkpoint).
- Command: `python ml_decoder/train_logical_cnn.py --basis X --resume <copy> --p-train 0.003,0.004,0.005 --hours 3 --batch 256 --p-report 0.0005,0.001,0.002,0.003,0.004,0.005 --out runs\x_3h_w32_resume.model_X.pt`
- Fresh AdamW, warm-up and cosine schedule. The clock curriculum runs within {0.003, 0.004} until 20% (36 minutes), then over all three.
- Started 20:27:45 CST. Shares the GPU with 3A, and the CPU with the PriorTilt decode processes, which run at BelowNormal priority.

## x_1h_w64 (run 3A): width 64, 1 hour, from scratch

- Commit: 0eab7dd (clean).
- Command: `python ml_decoder/train_logical_cnn.py --basis X --hours 1 --width 64 --depth 2 --batch 256 --p-report 0.0005,0.001,0.002,0.003,0.004,0.005 --out runs\x_1h_w64.model_X.pt`
- 1.0 M parameters. Started 20:28:26 CST. Pass line as for run 2, at p = 0.003.

Runs 3A and 3B share the GPU (96-97% busy together). From 20:35 CST both trainers run at High priority and the CPU decoders at BelowNormal on logical CPUs 0-11.

| Run | Time | M shots | Shots/s | Loss | p=0.003 bits right (always-0) | p=0.003 shots wrong | p=0.004 bits right | p=0.005 bits right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3B resume | 0.17 h | 10.1 | ~16,800 | 0.6236 | 66.36% (50.49%) | 91.63% | 54.83% (49.80%) | 51.01% (50.16%) |
| 3A w64 | 0.17 h | 8.0 | ~13,300 | 0.4098 | 61.32% (50.42%) | 96.53% | 52.63% (49.96%) | 50.17% (50.04%) |
| 3B resume | 0.33 h | 20.0 | ~16,500 | 0.6174 | 66.81% (50.45%) | 91.24% | 54.95% (50.21%) | 50.97% (50.15%) |
| 3A w64 | 0.33 h | 16.0 | ~13,300 | 0.6480 | 66.14% (50.61%) | 92.21% | 54.60% (50.15%) | 51.03% (50.12%) |
- 3B **stopped** at 20:50:52 CST on the orchestrator's decision: the cascade network replaces it. Wall 1,386.1 s, exit -1 (killed), stderr empty.
- Kept: `runs\x_3h_w32_resume.*` and `runs\x_3h_w32_resume.partial_0.33h.model_X.pt` (2,848,391 bytes, sha256 8c7cb13fb7f0b5800fe167440c81cf3e08bc99b749b7aeef1ffb42bb14f86b33).

## cascade_X_s1: structured (Cascade-style) network, X memory, all 1,728 detectors

- Code: `repo-cascade`, branch cascade, commit f0a95fa938d7e75a8911f399f9a26b9f27083ca6 (adds neural/ only).
- Command, from repo-cascade\challenge-1-ml-decoder-kit: `python -m neural.train --out C:\Users\salav\Salavat\hackMidwest\runs\cascade_X_s1 --seed 1 --max-hours 2`
- Author's settings: width 128, depth 8, 0.89 M parameters, batch 128, fp16. Step curriculum: 4,000 steps at p = 0.001, then a climb to 0.0055. Reports at step 0 and every 2,000 steps, on 2,000 error-model shots per p.
- Started 20:52:19 CST. Trainer at High priority. Shares the GPU with 3A until about 21:28.
- Its own log and checkpoint: `runs\cascade_X_s1\train.log` and `checkpoint.pt`.
- Learning check (orchestrator): the p=0.001 "shots wrong" must drop clearly below always-0 by steps 2,000-4,000. If it still equals always-0 at step 4,000, stop.
| 3A w64 | 0.50 h | 23.7 | ~12,800 | 0.6394 | 68.35% (50.57%) | 90.16% | 55.68% (50.06%) | 51.43% (49.91%) |
| 3A w64 | 0.67 h | 29.5 | ~9,700 | 0.6290 | 69.50% (50.52%) | 88.96% | 56.60% (49.91%) | 51.60% (50.12%) |

Cascade reports (EMA weights, 2,000 error-model shots per p; "shots wrong" with the always-0 value in brackets):

| Step | Training p | Loss | p=0.001 | p=0.002 | p=0.003 | p=0.004 | p=0.005 | Shots/s | Peak GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.001 | - | 0.9990 (0.9725) | 1.0000 (0.9985) | 1.0000 (1.0000) | 1.0000 (1.0000) | 0.9995 (0.9995) | - | - |
| 2000 | 0.001 | 0.1140 | 0.0385 (0.9725) | 0.2910 (0.9985) | 0.6620 (1.0000) | 0.9100 (1.0000) | 0.9855 (0.9995) | 271 (sharing with 3A) | 3.92 GB |

- Bits right at step 2000: 0.9946 / 0.9521 / 0.8382 / 0.6726 / 0.5545 at p = 0.001-0.005. The learning check is passed clearly.
- 3A **stopped** at 21:10:35 CST on the orchestrator's decision (the structured network gets the GPU). Wall 2,528.1 s, exit -1 (killed), stderr empty.
- Last line, 0.67 h: p = 0.003 at 69.50% bits right, 88.96% shots wrong.
- Kept: `runs\x_1h_w64.*` and `runs\x_1h_w64.partial_0.67h.model_X.pt` (4,121,681 bytes, sha256 8b15304fcc409afbffe3dbe976c16d448b53b50cb6bf310869950103503b37d0).
- Cascade s1 **crashed** at 21:10:43 CST with `CUDA error: an illegal memory access`. The cause was a display-driver reset (nvlddmkm Error 153 at 21:10:35/39/43, then Display 4101 "stopped responding and has successfully recovered" at 21:10:44), set off when 3A was force-killed while cascade shared the GPU.
- Checkpoint at step 2,000 kept: 14,661,708 bytes, sha256 f8478de3e3e60a3f092d757c94e7dd3c1eb003b2370913fa78118226c63c8ae4. Waiting for the orchestrator's go to resume.
| 4000 | 0.001 | 0.0161 | 0.0170 (0.9725) | 0.1650 (0.9985) | 0.5385 (1.0000) | 0.8525 (1.0000) | 0.9740 (0.9995) | 645 (alone, 0 decoders) | 3.92 GB |

- Resumed as cascade_X_s1_r1 at 21:12:44 CST from the step-2,000 checkpoint (backup `checkpoint.20261121-step2000.pt`, sha256 f8478de3...8ae4), alone on the GPU, `--max-hours 12`. Interpreter at High priority.
- From 21:20:39 CST, 5 CPU decode processes run beside it at BelowNormal on CPUs 0-11. Trainer: 1.00 CPU-s/s (1.13 with none). GPU 92% busy.
| 6000 | 0.001375 | 0.0206 | 0.0105 (0.9725) | 0.1070 (0.9985) | 0.4085 (1.0000) | 0.7415 (1.0000) | 0.9475 (0.9995) | 620 (5 decoders + a 15 s measure) | 3.92 GB |
| 8000 | 0.00175 | 0.0300 | 0.0055 (0.9725) | 0.0640 (0.9985) | 0.3130 (1.0000) | 0.6535 (1.0000) | 0.9025 (0.9995) | 622 (5 decoders) | 3.92 GB |
| 10000 | 0.002125 | 0.0390 | 0.0045 (0.9725) | 0.0420 (0.9985) | 0.2480 (1.0000) | 0.5790 (1.0000) | 0.8675 (0.9995) | 623 (5 decoders) | 3.92 GB |

- **Switched to commit 7ff6a68** at the step-10,000 boundary (21:40:44 CST). The trainer was alone on the GPU when stopped.
- Backup: `checkpoint.20261121-214046-step10000.pt`, sha256 97156713410eeaf913521156f2b0516dcb112d3815765c9acd7be2a7328506c7.
- Resumed 21:40:47 as cascade_X_s1_r2: `--steps 230000 --lr-decay-steps 230000 --report-shots 5000 --max-hours 11.532`, ending about 09:12:44 CST on 22 Nov. Fused AdamW runs on cuda.
- Stop only with the file `runs\cascade_X_s1\STOP`.
- `neural.measure --batch 32` beside the trainer, whole step: 121.30 ms at 2becd60, 101.15 ms at 7ff6a68. "Optimizer, clipping, moving average": 25.28 ms then 5.33 ms.

Cascade reports at 7ff6a68 (5,000 error-model shots per p, 95% interval +-0.014 or less; shots wrong with always-0 in brackets):

| Step | Training p | Loss | p=0.001 | p=0.002 | p=0.003 | p=0.004 | p=0.005 | Shots/s | GPU (tensors / held) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12000 | 0.0025 | 0.0574 | 0.0028 (0.9764) | 0.0386 (0.9990) | 0.2108 (1.0000) | 0.5390 (0.9996) | 0.8288 (0.9998) | 718 (5 decoders) | 3.93 / 5.95 GB |
| 14000 | 0.002875 | 0.0833 | 0.0020 (0.9764) | 0.0328 (0.9990) | 0.1898 (1.0000) | 0.5002 (0.9996) | 0.8056 (0.9998) | 724 (5 decoders) | 3.93 / 5.95 GB |
| 16000 | 0.00325 | 0.1131 | 0.0018 (0.9764) | 0.0306 (0.9990) | 0.1728 (1.0000) | 0.4768 (0.9996) | 0.7784 (0.9998) | 724 (5 decoders) | 3.93 / 5.95 GB |
| 18000 | 0.003625 | 0.1509 | 0.0014 (0.9764) | 0.0276 (0.9990) | 0.1706 (1.0000) | 0.4552 (0.9996) | 0.7522 (0.9998) | 727 (5 decoders) | 3.93 / 5.95 GB |
| 20000 | 0.004 | 0.1904 | 0.0018 (0.9764) | 0.0256 (0.9990) | 0.1554 (1.0000) | 0.4238 (0.9996) | 0.7274 (0.9998) | 724 (5 decoders) | 3.93 / 5.95 GB |

### Cascade step 20,000 on the kit's Stim shots (`neural.evaluate_stim` at 4d6ee8c, X memory)

- Snapshot `checkpoint_step20000.pt`, sha256 478423b5500659db079e40b7b37ccacea584fbccd6a92c607bf0dc0c08d612ef.
- Run one at a time beside the trainer, each only with at least 2 GiB of RAM available.

| File | Shots | Failures | Per shot [95% Wilson] | Top 50% by confidence: failures | Top 25% |
| --- | --- | --- | --- | --- | --- |
| p0.004 validation_X (seed 201) | 5,000 | 2,106 | 42.12% [40.76, 43.49] | 104 of 2,500 | 3 of 1,250 |
| p0.003 gen_X seed 9203 | 50,000 | 7,986 | 15.97% [15.65, 16.30] | 7 of 25,000 | 2 of 12,500 |
| p0.005 gen_X seed 9205 | 20,000 | 14,606 | 73.03% [72.41, 73.64] | 5,258 of 10,000 | 949 of 5,000 |

- Batch-one latency: about 18 ms mean on the GPU, measured beside the trainer.
- For comparison on p0.004 validation_X: relay-BP fails 31 and PriorTilt 26 of 5,000.
- Outputs: `runs\cascade_X_s1\eval\step20000_*`.
| 22000 | 0.004375 | 0.2445 | 0.0014 (0.9764) | 0.0220 (0.9990) | 0.1370 (1.0000) | 0.3962 (0.9996) | 0.6964 (0.9998) | 557 (window shared with 3 Stim scorings, about 189 s) | 3.93 / 5.95 GB |
| 24000 | 0.00475 | 0.3010 | 0.0008 (0.9764) | 0.0150 (0.9990) | 0.1004 (1.0000) | 0.3218 (0.9996) | 0.6256 (0.9998) | 726 (5 decoders) | 3.93 / 5.95 GB |
| 26000 | 0.005125 | 0.3614 | 0.0014 (0.9764) | 0.0150 (0.9990) | 0.0950 (1.0000) | 0.2886 (0.9996) | 0.5970 (0.9998) | 726 (5 decoders) | 3.93 / 5.95 GB |
| 28000 | 0.0055 | 0.4231 | 0.0016 (0.9764) | 0.0148 (0.9990) | 0.0862 (1.0000) | 0.2724 (0.9996) | 0.5724 (0.9998) | 726 (5 decoders) | 3.93 / 5.95 GB |
| 30000 | 0.0055 | 0.4254 | 0.0010 (0.9764) | 0.0140 (0.9990) | 0.0778 (1.0000) | 0.2588 (0.9996) | 0.5524 (0.9998) | 727 (5 decoders) | 3.93 / 5.95 GB |
| 32000 | 0.0055 | 0.4059 | 0.0006 (0.9764) | 0.0134 (0.9990) | 0.0762 (1.0000) | 0.2482 (0.9996) | 0.5382 (0.9998) | 523 (window shared with 4 Stim scorings, about 243 s) | 3.93 / 5.95 GB |

### Cascade step 30,000 on the kit's Stim shots (X memory)

- Snapshot sha256 d1f54ea15ef611f60f8e068046555a3c0c6ffb70167e77e857f91a95e20aecb9.
- Failures per shot: p0.003 seed 9203 8.31% (4,157 of 50,000); p0.004 validation_X 26.60% (1,330 of 5,000); p0.004 train_X 27.00% (5,399 of 20,000); p0.005 seed 9205 55.89% (11,177 of 20,000).
- As a selector on the step-8b candidates (X, 25,000 shots), "network breaks ties" fails 66 (relay-BP alone fails 122; the best 8b rule 68). See `runs\priortilt\results_network_select_step30000.txt`.

Artifacts pushed to branch `laptop-artifacts`, commit 0ef3ba0950eb5497c66769d2688016b0d6fbf591 (2026-11-21 22:52 CST): 54 files, 39.2 MB.
| 34000 | 0.0055 | 0.4064 | 0.0004 (0.9764) | 0.0144 (0.9990) | 0.0774 (1.0000) | 0.2456 (0.9996) | 0.5400 (0.9998) | 726 (5 decoders) | 3.93 / 5.95 GB |
| 36000 | 0.0055 | 0.3882 | 0.0008 (0.9764) | 0.0140 (0.9990) | 0.0718 (1.0000) | 0.2348 (0.9996) | 0.5248 (0.9998) | 728 (5 decoders) | 3.93 / 5.95 GB |
| 38000 | 0.0055 | 0.3814 | 0.0012 (0.9764) | 0.0148 (0.9990) | 0.0714 (1.0000) | 0.2276 (0.9996) | 0.5176 (0.9998) | 728 (5 decoders) | 3.93 / 5.95 GB |
| 40000 | 0.0055 | 0.3642 | 0.0008 (0.9764) | 0.0114 (0.9990) | 0.0624 (1.0000) | 0.2138 (0.9996) | 0.4838 (0.9998) | 728 (5 decoders) | 3.93 / 5.95 GB |
| 42000 | 0.0055 | 0.3567 | 0.0008 (0.9764) | 0.0084 (0.9990) | 0.0556 (1.0000) | 0.1952 (0.9996) | 0.4588 (0.9998) | 645 (window shared with 6 Stim scorings, about 217 s) | 3.93 / 5.95 GB |

### Cascade step 40,000 on six team X files (Stim shots)

- Snapshot sha256 e5e04f3cac0354e11f84c0e8f28770f8f829c3c4a622933b1ba69a29bca44973.
- p0.003 (from main 8f1e767): validation 6.16%, train 6.10%.
- p0.004: validation 21.86%, train 21.67%.
- p0.005 (from main 8f1e767): validation 47.54%, train 48.62%.
- As a selector, X p0.004 on 25,000 shots: "network breaks ties" fails 65.
- Pushed to `laptop-artifacts` as commit 39132e3. From now on `tools\snapshot_routine.ps1 -Step <N>` does this for each kept snapshot.
| 44000 | 0.0055 | 0.3496 | 0.0008 (0.9764) | 0.0066 (0.9990) | 0.0510 (1.0000) | 0.1770 (0.9996) | 0.4394 (0.9998) | 617 (5 decoders; GPU at its power cap, CPU at 68% of base clock) | 3.93 / 5.95 GB |
| 46000 | 0.0055 | 0.3406 | 0.0010 (0.9764) | 0.0068 (0.9990) | 0.0462 (1.0000) | 0.1658 (0.9996) | 0.4248 (0.9998) | 731 (decoders draining from 5) | 3.93 / 5.95 GB |
| 48000 | 0.0055 | 0.3357 | 0.0004 (0.9764) | 0.0056 (0.9990) | 0.0418 (1.0000) | 0.1570 (0.9996) | 0.4076 (0.9998) | 738 (3 decoders) | 3.93 / 5.95 GB |
| 50000 | 0.0055 | 0.3278 | 0.0006 (0.9764) | 0.0054 (0.9990) | 0.0390 (1.0000) | 0.1486 (0.9996) | 0.3888 (0.9998) | 739 (3 decoders) | 3.93 / 5.95 GB |

### Cascade step 50,000 on six team X files (Stim shots)

- Snapshot sha256 cf434ece2480e87cc45d1141366c1b90e274130f4939c7decfdf721e124a9efc.
- p0.003: validation 3.66%, train 3.96%.
- p0.004: validation 16.04%, train 15.43%.
- p0.005: validation 38.88%, train 39.14%.
- As a selector, X p0.004 on 25,000 shots: "network breaks ties" fails 63; weight plus network with C = 10 fails 62.
- Pushed as commit 68d7549.
| 51707 (STOP) | 0.0055 | 0.3229 | 0.0004 (0.9764) | 0.0058 (0.9990) | 0.0352 (1.0000) | 0.1370 (0.9996) | 0.3746 (0.9998) | 533 (window held the step-50k scorings) | 3.93 / 5.95 GB |

- **Shortened to 120,000 steps** (deadline Sunday morning).
- Stopped cleanly with the STOP file at step 51,707 (23:55:39 CST, exit 0). Backup `checkpoint.20261121-235540-step51707.pt`, sha256 ec761ba8613f940a89510ceb2bed14c0eb950c66e610009a3cab89ba03056eb8.
- Resumed 23:55:40 as cascade_X_s1_r3 with `--steps 120000 --lr-decay-steps 120000`. The learning rate drops from 8.92e-4 to 6.44e-4 on the new cosine. Expected end about 03:25-03:35 CST.

| 52000 | 0.0055 | 0.3170 | 0.0010 (0.9764) | 0.0056 (0.9990) | 0.0362 (1.0000) | 0.1340 (0.9996) | 0.3690 (0.9998) | 708 (3 decoders, 293 steps after the resume) | 3.93 / 5.95 GB |
| 54000 | 0.0055 | 0.3036 | 0.0006 (0.9764) | 0.0038 (0.9990) | 0.0326 (1.0000) | 0.1260 (0.9996) | 0.3586 (0.9998) | 721 (3 decoders) | 3.93 / 5.95 GB |
| 56000 | 0.0055 | 0.2940 | 0.0002 (0.9764) | 0.0044 (0.9990) | 0.0316 (1.0000) | 0.1188 (0.9996) | 0.3458 (0.9998) | 722 (3 decoders) | 3.93 / 5.95 GB |
| 58000 | 0.0055 | 0.2903 | 0.0004 (0.9764) | 0.0054 (0.9990) | 0.0308 (1.0000) | 0.1142 (0.9996) | 0.3390 (0.9998) | 725 (3 decoders) | 3.93 / 5.95 GB |
| 60000 | 0.0055 | 0.2836 | 0.0006 (0.9764) | 0.0040 (0.9990) | 0.0316 (1.0000) | 0.1134 (0.9996) | 0.3328 (0.9998) | 723 (3 decoders) | 3.93 / 5.95 GB |

### Cascade step 60,000 on six team X files

- Snapshot sha256 a6fccb2222fad6ef5d98cb5fe806ae0daf6bfbad57cbac454fa0b8f18d7e843f.
- p0.003: validation 2.70%, train 2.78%.
- p0.004: validation 12.32%, train 12.02%.
- p0.005: validation 32.92%, train 33.03%.
- As a selector: "network breaks ties" fails 62; weight plus network with C = 10 fails 59.
- Most confident 50% of the p0.004 shots: the network fails 6, relay-BP 9.
- Commit 874ae5e.
| 62000 | 0.0055 | 0.2859 | 0.0008 (0.9764) | 0.0028 (0.9990) | 0.0274 (1.0000) | 0.1048 (0.9996) | 0.3306 (0.9998) | 552 (window held the step-60k scorings) | 3.93 / 5.95 GB |
| 64000 | 0.0055 | 0.2752 | 0.0008 (0.9764) | 0.0022 (0.9990) | 0.0244 (1.0000) | 0.1020 (0.9996) | 0.3218 (0.9998) | 721 (3 decoders) | 3.93 / 5.95 GB |

### p = 0.005 team validation_X: R, P1, P2 (CPU)

- File: main 8f1e767, 5,000 shots.
- Failures: R 194, P1 137, P2 152. "P1 unless" 129 (537 ms/shot). Lowest weight among converged 109. "Network breaks ties" at step 60k: 126.
- A RAM dip to 506 MiB at 00:07:50 CST paused the runner, and two relay chunks were requeued. No result changed; the training was not affected.
- Commit 426d550.
| 66000 | 0.0055 | 0.2724 | 0.0002 (0.9764) | 0.0020 (0.9990) | 0.0206 (1.0000) | 0.0958 (0.9996) | 0.3054 (0.9998) | 720 (3 decoders) | 3.93 / 5.95 GB |
| 68000 | 0.0055 | 0.2708 | 0.0004 (0.9764) | 0.0024 (0.9990) | 0.0210 (1.0000) | 0.0976 (0.9996) | 0.2986 (0.9998) | 721 (3 decoders) | 3.93 / 5.95 GB |
