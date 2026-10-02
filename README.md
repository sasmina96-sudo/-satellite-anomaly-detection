# Satellite telemetry anomaly detection: FP32 versus INT8

Asmina S — M.Tech Space Technology

A reproducible experiment with a small LSTM autoencoder on **OPS-SAT-AD v2** telemetry. This repository records an accuracy/size/latency trade-off; it does not demonstrate compression or a substantial speed improvement from quantization.

## Recorded results

The held-out test set contains 529 segments: 416 normal and 113 anomalous. Each model uses its own validation-selected threshold, locked before the test evaluation.

| Metric | TFLite FP32 | TFLite INT8 |
|---|---:|---:|
| Precision | 0.919540 | 0.706422 |
| Recall | 0.707965 | 0.681416 |
| F1 | 0.800000 | 0.693694 |
| ROC-AUC | 0.910632 | 0.863491 |
| Average precision | 0.871781 | 0.786450 |
| False alarms (segments) | 7 | 32 |
| Missed anomalous segments | 33 | 36 |
| Model file size, bytes | 58,024 | 77,592 |
| Model file size, KiB | 56.66 | 75.77 |
| Mean runtime latency, µs/window | 48.936 | 49.067 |
| Median runtime latency, µs/window | 35.342 | 33.690 |
| P95 runtime latency, µs/window | 63.441 | 59.162 |

INT8 is approximately 33.7% larger, and its test F1 is approximately 0.1063 lower. The FP32/INT8 median latency ratio is 1.049×, while mean latency is essentially unchanged. These results apply to this model, conversion and CPU environment.

Timing used a Colab Intel Xeon CPU reported at 2.20 GHz, one interpreter thread, batch size one, 100 warmups and 3,072 measurements per model across six alternating-order repetitions. Each input is an 8-reading window. The timed region includes `set_tensor`, `invoke` and `get_tensor`, but excludes scaling, window creation, quantization, dequantization and anomaly scoring. It is not end-to-end processing latency or an on-board measurement.

## Method

- Data source: [OPS-SAT-AD v2, Zenodo record 15108715](https://zenodo.org/records/15108715). The notebook downloads `segments.csv` and `dataset.csv` and checks their recorded MD5 values. Raw telemetry is not bundled. Consult the source record for dataset attribution and reuse terms.
- 303,493 readings, 2,123 segments, 9 channels. Numeric `anomaly` is the target; the text `label` column is not a reliable normal/anomalous mapping.
- Preserve the provided train/test split. Hold out approximately 20% of development segments within each channel/class for validation using seed 42.
- Final partitions: training 1,015 normal + 253 anomalous segments; validation 258 normal + 68 anomalous; test 416 normal + 113 anomalous. Anomalous training segments are excluded from autoencoder fitting.
- Fit each channel's mean and standard deviation on normal training readings only.
- Pool univariate windows across channels: 8 readings × 1 feature, stride 4, include the final window without padding. Windows never cross segment boundaries. This is not a synchronized nine-feature model.
- LSTM(16) encoder, RepeatVector(8), LSTM(16) decoder, Dense(1): 3,281 trainable parameters. Both LSTMs use tanh and explicit unrolling.
- Adam learning rate 0.001, clipnorm 1, MSE loss, batch 128, up to 30 epochs; checkpoint by normal-validation reconstruction loss. The recorded best epoch was 30.
- Compare mean versus maximum window reconstruction MSE on validation. Select maximum. Select thresholds by maximum validation F1, choosing the highest threshold for exact ties.
- Freeze model weights before TFLite conversion. Calibrate INT8 with 967 normal training windows only. Enforce integer built-in operators and INT8 input/output; audit found no floating tensors.
- Score reconstructed INT8 output against the original scaled float input. Recalibrate each TFLite threshold on validation, then hash and lock both model files before test evaluation.

## Files

- `notebooks/Satellite_Anomaly_Detection_clean.ipynb`: cleaned Colab execution path for a **new** experiment. Outputs are deliberately empty; historical results are not presented as fresh execution.
- `archive/original_experiment.ipynb`: original uploaded notebook, unchanged, including the failed conversion and subsequent repair. Do not use its Run all path.
- `artifacts/models/lstm_fp32_20260929T185150250455Z/`: trained model, both TFLite files, calibration inputs, scores, protocol, training history and timing evidence.
- `artifacts/models/channel_scaling_v1.json`: original training-only scaler.
- `artifacts/results/`: original split, integrity checks, summary and environment record.
- `scripts/verify_artifacts.py`: verifies model hashes and sizes; recalculates test metrics from saved scores and latency summaries from raw timings.
- `VERIFICATION.md`: checks performed and remaining validation limits.

## Check the archived results

With NumPy, pandas and scikit-learn installed, run from the extracted repository:

```bash
python scripts/verify_artifacts.py
```

This requires no TensorFlow and does not run model inference. It checks consistency of saved evidence; it does not independently reproduce the original predictions.

## Run a new experiment in Colab

1. Upload `notebooks/Satellite_Anomaly_Detection_clean.ipynb` to Colab.
2. Use the versions recorded in `requirements.txt` (original Python: 3.13.15). Installing packages can require restarting the Colab runtime. `google.colab` is supplied by Colab, not by this requirements file.
3. Run the cells in order. Mount Drive when prompted. The notebook creates a timestamped directory under `MyDrive/Satellite_Anomaly_Reproductions`, preserving the archived experiment.
4. Review conversion numerical checks, integer tensor audit, validation thresholds and final output. Do not substitute the historical numbers for newly measured results.

The former F1-based run lookup has been removed: every downstream cell uses the run created by the training cell. The failed conversion path is replaced by the verified frozen-function approach. The notebook uses a TensorFlow internal freezing helper and the recorded TFLite interpreter API; compatibility with other TensorFlow versions is not established. A fixed seed does not guarantee bit-identical retraining across hardware/software environments.

## Limitations and appropriate claims

One training run and one Colab CPU environment were evaluated. No hardware deployment, peak RAM measurement, energy measurement or satellite qualification was performed. File size is not RAM usage. Integer tensors alone do not guarantee smaller serialized models or faster inference.

Validation labels influence scoring and thresholds; describe this as normal-only autoencoder training with supervised validation calibration, not fully unsupervised detection. Segment isolation prevents overlapping windows across these partitions, but temporal independence and possible related acquisitions across segments have not been established. Maximum segment error also depends on segment length. Results are segment-level, not point-level anomaly localization or real-time alert delay.

The validation-only squared-magnitude baseline is exploratory and was not included in the final locked test comparison. No multi-seed uncertainty estimates or external benchmark superiority claims are supported.

The earlier NASA SMAP draft numbers (3.44× compression, 2.02× speedup and 99.4% F1 retention) are not results of this experiment and must not be used to describe it.

## GitHub upload

Extract the ZIP locally. Upload the **contents** of the `satellite-anomaly-detection` directory into your repository so this README appears at its root. Preserve the folders. The ZIP itself is a delivery container, not the intended repository layout. Review the original notebook before public publication. No open-source code license has been selected in this package.
