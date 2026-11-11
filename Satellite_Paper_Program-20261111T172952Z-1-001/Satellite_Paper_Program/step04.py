# STEP 4 — OPS-SAT: representative INT8 inputs, check windows, fixed statistical controls
rng = np.random.default_rng(42)
eligible = (OPS_INFO["subset"].eq("train") & OPS_INFO["segment_anomaly"].eq(0)).to_numpy()
chosen = []
for ch in sorted(OPS_INFO.loc[eligible, "channel"].unique()):
    idx = np.flatnonzero(eligible & OPS_INFO["channel"].eq(ch).to_numpy())
    chosen.extend(rng.choice(idx, size=min(128, len(idx)), replace=False).tolist())
    cx = OPS_X[idx]
    chosen.extend([int(idx[np.argmin(cx.min(axis=(1, 2)))]), int(idx[np.argmax(cx.max(axis=(1, 2)))])])
OPS_REP = OPS_X[np.unique(chosen)].astype(np.float32)
check("OPS-SAT representative INT8 inputs", len(OPS_REP), N_REPRESENTATIVE)
np.savez_compressed(OUT / "data/opssat_representative_inputs.npz", X=OPS_REP)

val_idx = np.flatnonzero(OPS_INFO["subset"].eq("validation").to_numpy())
OPS_X_CHECK = OPS_X[np.random.default_rng(42).choice(val_idx, size=min(256, len(val_idx)), replace=False)]
OPS_X_BENCH = OPS_X[np.random.default_rng(42).choice(val_idx, size=min(512, len(val_idx)), replace=False)]


def ops_segment_scores(window_scores):
    """Segment score = maximum window reconstruction error."""
    df = OPS_VAL_INFO.copy()
    df["score"] = window_scores
    seg = df.groupby(["channel", "segment"], sort=True).agg(label=("segment_anomaly", "first"),
                                                            score=("score", "max"))
    return seg["label"].to_numpy().astype(int), seg["score"].to_numpy()


OPS_BASE = Baselines.fit(OPS_X_FIT)
OPS_BASE.save(OUT / "models/opssat/baselines_pca4.npz")
for name, w in OPS_BASE.scores(OPS_X_VAL).items():
    y, s = ops_segment_scores(w)
    for cond, t in {"f1": max_f1_threshold(y, s), "p99": p99_threshold(s[y == 0])}.items():
        ROWS.append(dict(cohort="OPS-SAT validation", target="annotated anomaly", model=name, kind="baseline",
                         seed=np.nan, condition=cond, **evaluate(y, s, t)))
check("Scored validation segments", len(y), 326)
save_progress()
print(pd.DataFrame(ROWS)[["model", "condition", "f1", "precision", "recall"]].round(4).to_string(index=False))
