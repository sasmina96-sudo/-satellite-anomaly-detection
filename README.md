# Edge AI Optimization for Satellite Telemetry Anomaly Detection

Code and results for the paper

**"Edge AI Optimization for Satellite Telemetry Anomaly Detection: INT8 Autoencoders, Energy Estimates and Downlink Savings on OPS-SAT and ESA Mission 2"**
Asmina S, Independent Researcher, Kerala, India
Preprint: https://doi.org/10.5281/zenodo.24143856
The study compares compact LSTM (3,281 parameters) and dense (428 parameters) telemetry autoencoders in FP32 and fully integer INT8 form (LiteRT), and asks what onboard anomaly detection saves in power and downlink.

## Main results (run `20261108T181639Z`)

| Question | Result |
|---|---|
| Does INT8 change detection? | OPS-SAT LSTM F1 0.7962 → 0.7085; dense 0.8049 → 0.8015. ESA Mission 2: no loss (LSTM 0.3249 → 0.3309, dense 0.3178 → 0.3314) |
| Estimated energy per inference | LSTM 338 → 65 nJ with INT8 (about 5.2x); measured CPU latency improves only 1.11x |
| Downlink saved by onboard filtering (policy P1) | about 98.1% of the raw volume of one year of ESA Mission 2 telemetry (11 channels) |
| Rare events still transmitted | 94.5–96.0% of 200 annotated rare events; random flags at the same budget capture 11–13% |
| After lossless compression (LZMA2, 2.7x) | P1 is still about 95.9% below the compressed stream |
| Learned models vs simple controls | ZScore and PCA4 controls give the same saving and similar event capture |

All energy and cost figures are model-based estimates, not hardware measurements. The ESA confirmation year contains a single true anomaly.

## Repository contents

| Path | Content |
|---|---|
| `Satellite_Paper_Program-…/` | Pipeline code: `RUN_Satellite_Paper.ipynb` (runner), `step01.py` … `step14.py`, `RUN_Extra_Analysis.ipynb`, `extra_analysis.py` |
| `PAPER_NUMBERS.md` | Every table of the reported run |
| `all_metrics.csv`, `esa_per_channel.csv`, `esa_cohort_description.csv`, `artifacts.csv`, `environment.json` | Detection metrics, per-channel results, exported-file inventory, software versions |
| `esa_confirmation_predictions.csv.gz` | Scores and flags for all 72,270 confirmation channel-blocks |
| `power_estimate.csv` | Energy estimate per inference |
| `downlink_policy_per_run.csv`, `downlink_policy_summary.csv`, `downlink_cost_per_year.csv` | Downlink policy and link accounting |
| `extra/` | Flag-budget sweep, random-flag baseline, lossless-compression baseline (`EXTRA_NUMBERS.md`) |
| `lock/` | Lock record (SHA-256 hashes) written before confirmation scoring |
| `models/` | Exported FP32 and INT8 LiteRT models, checkpoints and calibration scores |
| `figures/` | Figures of the run |
| `archive/`, `artifacts/`, `notebooks/`, `scripts/`, `SHA256SUMS.json`, `VERIFICATION.md` | Earlier OPS-SAT study (archived test reproduction) |

## How to reproduce

Everything runs in Google Colab on CPU; no hardware is required.

1. Copy the pipeline folder to Google Drive as `MyDrive/Satellite_Paper_Program`.
2. Open `RUN_Satellite_Paper.ipynb` in Colab and choose **Runtime → Run all**. The notebook downloads OPS-SAT-AD and the eleven ESA Mission 2 channels, trains the models, exports FP32/INT8, locks the protocol and scores the confirmation year. Results are written to `MyDrive/Satellite_Full_Paper_Run/runs/<timestamp>/`.
3. Open `RUN_Extra_Analysis.ipynb` and choose **Run all** for the flag-budget and compression analysis (reads the finished run; nothing is retrained).

Software of the reported run: Python 3.13, TensorFlow 2.20.0, Keras 3.13.2, LiteRT 2.3.0.

## Data

Raw data are not redistributed here. They are downloaded by the notebook from the public records:

- OPSSAT-AD, version 2 — https://doi.org/10.5281/zenodo.15108715
- ESA Anomaly Dataset, version 2 — https://doi.org/10.5281/zenodo.15237121

## Contact

Asmina S — s.asmina96@gmail.com
