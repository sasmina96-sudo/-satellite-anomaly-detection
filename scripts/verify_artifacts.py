"""Verify recorded files and recompute metrics; does not retrain or run inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/models/lstm_fp32_20260929T185150250455Z'
protocol = json.loads((RUN / 'evaluation_protocol_v1.json').read_text())
reports = {r['model']: r for r in json.loads((RUN / 'final_test_metrics.json').read_text())}
splits = pd.read_csv(ROOT / 'artifacts/results/segment_split_v1.csv')
assert not splits.duplicated(['channel', 'segment']).any()
expected = splits.loc[splits.subset.eq('test'), ['channel', 'segment', 'anomaly']]
assert len(expected) == 529
results = []
for item in protocol['models']:
    data = (RUN / item['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    assert len(data) == item['size_bytes']
    suffix = 'int8' if 'INT8' in item['model'] else 'fp32'
    scores = pd.read_csv(RUN / f'test_{suffix}_segment_scores.csv', float_precision='round_trip')
    assert len(scores) == 529
    merged = expected.merge(scores, on=['channel', 'segment'], validate='one_to_one', suffixes=('_expected', ''))
    assert len(merged) == 529 and merged.anomaly_expected.eq(merged.anomaly).all()
    y, score = scores.anomaly.to_numpy(), scores.score.to_numpy()
    pred = (score >= item['threshold']).astype(int)
    assert np.array_equal(pred, scores.prediction.to_numpy())
    report = reports[item['model']]
    assert report['model_sha256'] == item['sha256']
    assert report['threshold'] == item['threshold']
    values = {
        'precision': precision_score(y, pred), 'recall': recall_score(y, pred),
        'f1': f1_score(y, pred), 'roc_auc': roc_auc_score(y, score),
        'average_precision': average_precision_score(y, score),
    }
    for name, value in values.items():
        assert np.isclose(value, report[name], rtol=0, atol=1e-12), name
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    for name, value in zip(['true_negatives', 'false_alarms', 'missed', 'detected'], [tn, fp, fn, tp]):
        assert report[name] == int(value)
    results.append({'model': item['model'], 'size_bytes': len(data), **values})
raw = pd.read_csv(RUN / 'latency_raw.csv', float_precision='round_trip')
summary = pd.read_csv(RUN / 'latency_summary.csv', float_precision='round_trip').set_index('model')
for name, group in raw.groupby('model'):
    v = group.latency_us
    assert len(v) == int(summary.loc[name, 'measurements']) == 3072
    for metric, actual in [('mean_us', v.mean()), ('median_us', v.median()), ('p95_us', v.quantile(.95))]:
        assert np.isclose(actual, summary.loc[name, metric], rtol=0, atol=1e-10)
print('PASS: model SHA-256 hashes and sizes match the locked protocol.')
print('PASS: all 529 test segment labels, predictions, metrics and confusion matrices match.')
print('PASS: latency summaries match all 6,144 recorded measurements.')
print(pd.DataFrame(results).to_string(index=False))
print('This is an artifact consistency check, not fresh inference or training.')
