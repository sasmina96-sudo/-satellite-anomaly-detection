# STEP 10 — ESA CONFIRMATION: verify the lock, then score 2001-10 → 2002-10 exactly once
lock = json.loads(LOCK_PATH.read_text())
changed = [p for p, h in lock["files"].items() if sha256_file(OUT / p) != h]
if changed or code_fingerprint(EVALUATOR_OBJS) != lock["evaluator_code_sha256"]:
    raise RuntimeError(f"Lock verification FAILED: {changed or 'evaluator code changed'}")
print(f"✓ Lock verified: {len(lock['files'])} files and evaluator code unchanged")
THR = json.loads((OUT / "lock/esa_thresholds.json").read_text())

ESA_X_CONF = ESA_X[CONF_ROWS]
conf = ESA.iloc[CONF_ROWS].reset_index(drop=True)
Y_PRIMARY = conf["primary"].to_numpy().astype(int)
KEEP_SECONDARY = ~conf["rare_only"].to_numpy()                   # secondary: rare-only blocks excluded
TARGETS = {ESA_PRIMARY: (np.ones(len(conf), dtype=bool), Y_PRIMARY),
           ESA_SECONDARY: (KEEP_SECONDARY, conf["anomaly"].to_numpy()[KEEP_SECONDARY].astype(int))}
PRED = conf[["channel", "block", "start", "raw_readings", "primary", "anomaly", "rare_only"]].copy()
art = {(a["cohort"], a["arch"], a["seed"]): a for a in ARTIFACTS}

for arch in ARCHS:
    for seed in SEEDS:
        key, run_dir = f"{arch}_seed{seed}", OUT / "models/esa" / f"{arch}_seed{seed}"
        scores = {}
        for fmt in ("fp32", "int8"):
            recon, clipped = Runner((run_dir / f"model_{fmt}.tflite").read_bytes()).run(ESA_X_CONF)
            scores[fmt] = window_mse(ESA_X_CONF, recon)
            PRED[f"{key}_{fmt}_score"] = scores[fmt]
            PRED[f"{key}_{fmt}_flag"] = scores[fmt] >= THR[key][f"{fmt}_f1"]
            art[("ESA", arch, seed)][f"{fmt}_clip_fraction"] = float(clipped.sum() / ESA_X_CONF.size)
        for target, (mask, y) in TARGETS.items():
            for cond, (fmt, t) in conditions(THR[key]).items():
                ROWS.append(dict(cohort="ESA confirmation", target=target, model=arch, kind="learned",
                                 seed=seed, condition=cond, **evaluate(y, scores[fmt][mask], t)))
        f1 = {r["condition"]: r for r in ROWS[-10:-5]}
        print(f"  {NAME[arch]:5} seed {seed}: primary F1 FP32 {f1['fp32_f1']['f1']:.4f} → INT8 {f1['int8_f1']['f1']:.4f} | "
              f"precision {f1['fp32_f1']['precision']:.3f}/{f1['int8_f1']['precision']:.3f} | "
              f"recall {f1['fp32_f1']['recall']:.3f}/{f1['int8_f1']['recall']:.3f}")

for name, s in Baselines.load(OUT / "models/esa/baselines_pca4.npz").scores(ESA_X_CONF).items():
    PRED[f"{name}_score"], PRED[f"{name}_flag"] = s, s >= THR[name]["f1"]
    for target, (mask, y) in TARGETS.items():
        for cond in ("f1", "p99"):
            ROWS.append(dict(cohort="ESA confirmation", target=target, model=name, kind="baseline",
                             seed=np.nan, condition=cond, **evaluate(y, s[mask], THR[name][cond])))

# Per-channel results (primary target, own max-F1 thresholds)
per_channel = []
for arch in ARCHS:
    for seed in SEEDS:
        for fmt in ("fp32", "int8"):
            for ch, g in PRED.groupby("channel"):
                m = evaluate(g["primary"].astype(int), g[f"{arch}_seed{seed}_{fmt}_score"], THR[f"{arch}_seed{seed}"][f"{fmt}_f1"])
                per_channel.append(dict(model=arch, seed=seed, format=fmt, channel=ch,
                                        **{k: m[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}))
pd.DataFrame(per_channel).to_csv(OUT / "results/esa_per_channel.csv", index=False)

pos_in_conf = pd.Series(np.arange(len(CONF_ROWS)), index=CONF_ROWS)
CONF_EVENTS = ESA_EVENTS[ESA_EVENTS["row"].isin(CONF_ROWS)].copy()
CONF_EVENTS["pos"] = pos_in_conf.loc[CONF_EVENTS["row"]].to_numpy()
PRED.to_csv(OUT / "results/esa_confirmation_predictions.csv.gz", index=False)
save_progress()
print("\nESA confirmation scored once. Predictions saved.")
