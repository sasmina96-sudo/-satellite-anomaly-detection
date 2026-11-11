# Results for run 20261108T181639Z

Environment: Python 3.13.18, TensorFlow 2.20.0, LiteRT 2.3.0, CPU: Intel(R) Xeon(R) CPU @ 2.20GHz

All numbers below come from this run and replace the numbers in the current draft.

## Table 1 — Mean F1 ± sample SD over seeds (primary threshold rule)

| Cohort | Model | FP32 F1 | INT8 F1 | Mean change | SD of change |
|---|---|---|---|---|---|
| OPS-SAT validation | LSTM | 0.7962 +/- 0.0189 | 0.7085 +/- 0.0324 | -0.0877 | 0.0175 |
| OPS-SAT validation | Dense | 0.8049 +/- 0.0188 | 0.8015 +/- 0.0220 | -0.0034 | 0.0042 |
| ESA confirmation | LSTM | 0.3249 +/- 0.0178 | 0.3309 +/- 0.0080 | +0.0060 | 0.0144 |
| ESA confirmation | Dense | 0.3178 +/- 0.0217 | 0.3314 +/- 0.0234 | +0.0137 | 0.0019 |

## Table 2 — Every training seed (primary threshold rule)

| Cohort | Model | Seed | FP32 F1 | INT8 F1 | Change |
|---|---|---|---|---|---|
| OPS-SAT validation | LSTM | 42 | 0.7748 | 0.6724 | -0.1024 |
| OPS-SAT validation | LSTM | 43 | 0.8034 | 0.7350 | -0.0684 |
| OPS-SAT validation | LSTM | 44 | 0.8103 | 0.7179 | -0.0924 |
| OPS-SAT validation | Dense | 42 | 0.7869 | 0.7788 | -0.0081 |
| OPS-SAT validation | Dense | 43 | 0.8034 | 0.8033 | -0.0001 |
| OPS-SAT validation | Dense | 44 | 0.8244 | 0.8226 | -0.0018 |
| ESA confirmation | LSTM | 42 | 0.3098 | 0.3219 | +0.0121 |
| ESA confirmation | LSTM | 43 | 0.3445 | 0.3340 | -0.0105 |
| ESA confirmation | LSTM | 44 | 0.3205 | 0.3369 | +0.0164 |
| ESA confirmation | Dense | 42 | 0.3379 | 0.3526 | +0.0147 |
| ESA confirmation | Dense | 43 | 0.2948 | 0.3063 | +0.0114 |
| ESA confirmation | Dense | 44 | 0.3205 | 0.3354 | +0.0148 |

## Table 3 — Fixed controls, separately calibrated max-F1 thresholds

| Cohort | Baseline | F1 | F1 (nominal P99) |
|---|---|---|---|
| OPS-SAT validation | PCA4 | 0.8226 | 0.7863 |
| OPS-SAT validation | ZScore | 0.3452 | 0.0278 |
| OPS-SAT validation | Persistence | 0.5532 | 0.4894 |
| ESA confirmation | PCA4 | 0.3221 | 0.3538 |
| ESA confirmation | ZScore | 0.3558 | 0.4229 |
| ESA confirmation | Persistence | 0.3192 | 0.3453 |

## Table 4 — Mean F1 under every declared operating condition

| Cohort | Model | FP32, own max-F1 threshold | INT8, own max-F1 threshold | INT8, FP32 threshold transferred | FP32, nominal P99 | INT8, nominal P99 |
|---|---|---|---|---|---|---|
| OPS-SAT validation | LSTM | 0.7962 | 0.7085 | 0.5343 | 0.7825 | 0.6714 |
| OPS-SAT validation | Dense | 0.8049 | 0.8015 | 0.7823 | 0.7828 | 0.7827 |
| ESA confirmation | LSTM | 0.3249 | 0.3309 | 0.3370 | 0.3528 | 0.3326 |
| ESA confirmation | Dense | 0.3178 | 0.3314 | 0.3308 | 0.3515 | 0.3724 |

## Table 5 — ESA confirmation: precision, recall and ranking (mean over seeds, primary target)

| Model | Format | Precision | Recall | F1 | Roc Auc | Average Precision |
|---|---|---|---|---|---|---|
| LSTM | FP32 | 0.9550 | 0.1959 | 0.3249 | 0.6161 | 0.3042 |
| LSTM | INT8 | 0.9553 | 0.2002 | 0.3309 | 0.6043 | 0.2914 |
| Dense | FP32 | 0.9499 | 0.1911 | 0.3178 | 0.6340 | 0.3093 |
| Dense | INT8 | 0.9523 | 0.2010 | 0.3314 | 0.6366 | 0.3271 |

## Table 6 — ESA secondary target (anomaly only; rare-only blocks excluded) — descriptive

| Model | Condition | Mean F1 | Mean recall |
|---|---|---|---|
| LSTM | FP32, own max-F1 threshold | 0.0311 | 0.1667 |
| LSTM | INT8, own max-F1 threshold | 0.0659 | 0.2500 |
| Dense | FP32, own max-F1 threshold | 0.0304 | 0.1667 |
| Dense | INT8, own max-F1 threshold | 0.0304 | 0.1667 |
| PCA4 | baseline, max-F1 | 0.0606 | 0.2500 |
| ZScore | baseline, max-F1 | 0.1538 | 0.5000 |
| Persistence | baseline, max-F1 | 0.0290 | 0.2500 |

## Table 7 — Exported files, tensor audits, CPU latency (batch 1, 1 thread) and clipping

| Cohort | Model | FP32 file (bytes) | INT8 file (bytes) | INT8 floating tensors | Default-quantizer floating tensors (diagnostic) | Median latency FP32 / INT8 (µs) | INT8 input clipping |
|---|---|---|---|---|---|---|---|
| OPS-SAT | LSTM | 58,876 | 76,176, 76,336 | 0 | 0, 0, 5 | 35.1 / 33.9 | 0.000% |
| OPS-SAT | Dense | 4,140 | 5,112 | 0 | 0, 0, 0 | 3.1 / 3.0 | 0.000% |
| ESA | LSTM | 58,876 | 75,976, 76,336 | 0 | 0, 0, 0 | 37.3 / 33.7 | 0.331% |
| ESA | Dense | 4,140 | 5,112 | 0 | 0, 0, 0 | 3.1 / 3.0 | 0.331% |

## ESA cohort description

| Quantity | Value |
|---|---|
| Nominal fitting blocks available | 102,085 |
| Fitting windows used | 29,242 |
| Calibration blocks | 18,216 |
| Nominal calibration blocks | 17,876 |
| Confirmation channel-blocks | 72,270 |
| Positive confirmation blocks | 3,275 |
| Rare-only positive blocks | 3,271 |
| Blocks with a true-anomaly annotation | 4 |
| Distinct rare-event IDs | 200 |
| Distinct true-anomaly IDs | 1 |
| Non-finite blocks excluded (all periods) | 11 |

## Power — energy per inference (ESTIMATE, not measured)

| Model | Format | MACs (graph) | MACs (analytic) | Arithmetic (nJ) | Weight fetch (nJ) | Estimate total (nJ) | Per day, 198 inferences (µJ) | Median CPU latency (µs) | Latency × 1 W (µJ) |
|---|---|---|---|---|---|---|---|---|---|
| LSTM | FP32 | 23168 | 25216 | 106.57 | 231.68 | 338.25 | 66.97 | 37.29 | 37.29 |
| LSTM | INT8 | 23168 | 25216 | 6.95 | 57.92 | 64.87 | 12.84 | 33.66 | 33.66 |
| Dense | FP32 | 384 | 384 | 1.77 | 3.84 | 5.61 | 1.11 | 3.06 | 3.06 |
| Dense | INT8 | 384 | 384 | 0.12 | 0.96 | 1.08 | 0.21 | 3.03 | 3.03 |

Arithmetic + weight-fetch estimate uses Horowitz (ISSCC 2014) 45 nm per-operation energies; it excludes activations, requantization, control, idle power and interpreter overhead. Latency-based column assumes 1 W active power on the Colab CPU.

## Cost — downlink policies on the ESA confirmation year

| Detector | Flagged blocks | Block recall | Rare-event recall (by ID) | True anomalies caught | P1 saving % | P2 saving % | False alarms in sent raw % |
|---|---|---|---|---|---|---|---|
| LSTM FP32 | 672.667 | 0.196 | 0.953 | 0/1 · 1/1 · 1/1 | 98.113 | 99.069 | 4.498 |
| LSTM INT8 | 687.0 | 0.2 | 0.955 | 1/1 · 1/1 · 1/1 | 98.093 | 99.05 | 4.467 |
| Dense FP32 | 661.333 | 0.191 | 0.945 | 1/1 · 0/1 · 1/1 | 98.128 | 99.085 | 5.007 |
| Dense INT8 | 693.667 | 0.201 | 0.948 | 1/1 · 0/1 · 1/1 | 98.084 | 99.04 | 4.773 |
| PCA4 (control) | 662.0 | 0.194 | 0.955 | 1/1 | 98.127 | 99.084 | 4.228 |
| ZScore (control) | 733.0 | 0.218 | 0.96 | 1/1 | 98.029 | 98.986 | 2.729 |
| Persistence (control) | 698.0 | 0.194 | 0.96 | 1/1 | 98.078 | 99.034 | 9.168 |
| Oracle (true labels) | 3275.0 | 1.0 | 1.0 | 1/1 | 94.513 | 95.47 | 0.0 |

| Detector | Link | Bits saved / year (Mbit) | Transmit time saved / year (s) | Transmit energy saved / year (J) | Ground-station minutes saved / year | Minutes saved / year at 1,000 channels |
|---|---|---|---|---|---|---|
| LSTM FP32 | 4.30 Mbit/s (datasheet maximum) | 363.147 | 84.453 | 1097.886 | 1.408 | 127.959 |
| LSTM FP32 | 0.54 Mbit/s (lowest symbol rate, scaled) | 363.147 | 675.622 | 8783.089 | 11.26 | 1023.67 |
| LSTM INT8 | 4.30 Mbit/s (datasheet maximum) | 363.074 | 84.436 | 1097.664 | 1.407 | 127.933 |
| LSTM INT8 | 0.54 Mbit/s (lowest symbol rate, scaled) | 363.074 | 675.486 | 8781.314 | 11.258 | 1023.463 |
| Dense FP32 | 4.30 Mbit/s (datasheet maximum) | 363.205 | 84.466 | 1098.062 | 1.408 | 127.979 |
| Dense FP32 | 0.54 Mbit/s (lowest symbol rate, scaled) | 363.205 | 675.73 | 8784.495 | 11.262 | 1023.834 |
| Dense INT8 | 4.30 Mbit/s (datasheet maximum) | 363.04 | 84.428 | 1097.561 | 1.407 | 127.921 |
| Dense INT8 | 0.54 Mbit/s (lowest symbol rate, scaled) | 363.04 | 675.422 | 8780.491 | 11.257 | 1023.367 |
| PCA4 (control) | 4.30 Mbit/s (datasheet maximum) | 363.202 | 84.466 | 1098.052 | 1.408 | 127.978 |
| PCA4 (control) | 0.54 Mbit/s (lowest symbol rate, scaled) | 363.202 | 675.724 | 8784.413 | 11.262 | 1023.824 |
| ZScore (control) | 4.30 Mbit/s (datasheet maximum) | 362.838 | 84.381 | 1096.952 | 1.406 | 127.85 |
| ZScore (control) | 0.54 Mbit/s (lowest symbol rate, scaled) | 362.838 | 675.048 | 8775.62 | 11.251 | 1022.799 |
| Persistence (control) | 4.30 Mbit/s (datasheet maximum) | 363.017 | 84.423 | 1097.494 | 1.407 | 127.913 |
| Persistence (control) | 0.54 Mbit/s (lowest symbol rate, scaled) | 363.017 | 675.381 | 8779.951 | 11.256 | 1023.304 |
| Oracle (true labels) | 4.30 Mbit/s (datasheet maximum) | 349.823 | 81.354 | 1057.604 | 1.356 | 123.264 |
| Oracle (true labels) | 0.54 Mbit/s (lowest symbol rate, scaled) | 349.823 | 650.833 | 8460.83 | 10.847 | 986.111 |

Assumptions: 16-bit readings; summary = 49 bits/block; transmitter DC power 13.0 W; link information rates 4.30 Mbit/s (datasheet maximum), 0.54 Mbit/s (lowest symbol rate, scaled); headers, timestamps, compression and retransmissions not modelled; ground-station price not applied (minutes reported).

## Lock

Lock file SHA-256: 71a655d821af86f9c61066628959c20fb32a9a3e1078568c16c4e838a20d2aba (30 files hashed).