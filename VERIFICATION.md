# Verification record

Prepared from the notebook and ZIP provided on 2026-10-02.

Completed:

- SHA-256 and byte sizes of both TFLite files match `evaluation_protocol_v1.json`.
- All 529 saved test segment keys and labels match the recorded test partition.
- Predictions recomputed from saved scores and locked thresholds match both prediction files.
- Precision, recall, F1, ROC-AUC, average precision and all confusion matrix counts match the saved JSON within numerical tolerance.
- Mean, median and P95 latency match the 6,144 raw measurements.
- Every clean notebook code cell parses as Python; failed conversion/diagnostic cells and the historical F1-based run lookup have been removed from its execution path.
- The original notebook is retained byte-for-byte. Original artifact bytes are retained unchanged.

Not performed:

- Fresh TensorFlow inference or end-to-end execution of the cleaned notebook: TensorFlow is not installed in the packaging environment.
- Retraining, dataset re-download or fresh hardware benchmarking.
- Independent verification of dataset provenance, temporal independence or dataset license terms.

Thus the package has verified artifact consistency and static code checks, not a newly executed full reproduction. Run the clean notebook in the recorded Colab environment to complete that gate. Do not use a fresh result to retroactively modify the archived evaluation protocol.
