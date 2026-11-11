# STEP 9 — ESA: train 2 × 3 models, calibrate thresholds, then LOCK (no confirmation scoring yet)
ESA_X_FIT, ESA_X_CAL, ESA_X_CAL_NOM = ESA_X[FIT_ROWS], ESA_X[CAL_ROWS], ESA_X[CAL_NOMINAL_ROWS]
CAL_Y = ESA["primary"].to_numpy()[CAL_ROWS].astype(int)
ESA_REP = ESA_X[esa_representative_rows(FIT_ROWS)]
check("ESA representative INT8 inputs", len(ESA_REP), N_REPRESENTATIVE)
np.savez_compressed(OUT / "data/esa_representative_inputs.npz", X=ESA_REP)
ESA_X_CHECK = ESA_X_CAL[np.random.default_rng(42).choice(len(CAL_ROWS), size=256, replace=False)]
ESA_X_BENCH = ESA_X_CAL[np.random.default_rng(42).choice(len(CAL_ROWS), size=512, replace=False)]

Baselines.fit(ESA_X_FIT).save(OUT / "models/esa/baselines_pca4.npz")
LOCKED_THRESHOLDS = {}
for name, s in Baselines.load(OUT / "models/esa/baselines_pca4.npz").scores(ESA_X_CAL).items():
    LOCKED_THRESHOLDS[name] = {"f1": max_f1_threshold(CAL_Y, s), "p99": p99_threshold(s[CAL_Y == 0])}

for arch in ARCHS:
    for seed in SEEDS:
        run_dir = OUT / "models/esa" / f"{arch}_seed{seed}"
        print(f"\n=== ESA | {NAME[arch]} | seed {seed} ===")
        try:
            files, info = train_and_export(arch, seed, "ESA", ESA_X_FIT, ESA_X_CAL_NOM, ESA_REP, ESA_X_CHECK, run_dir)
            cal = {fmt: window_mse(ESA_X_CAL, Runner(files[fmt]).run(ESA_X_CAL)[0]) for fmt in ("fp32", "int8")}
            thr = threshold_set(CAL_Y, cal)
            LOCKED_THRESHOLDS[f"{arch}_seed{seed}"] = thr
            np.savez_compressed(run_dir / "calibration_scores.npz", y=CAL_Y, **cal)
            for fmt, timing in latency_benchmark(files, ESA_X_BENCH).items():
                info.update({f"{fmt}_{k}": float(v) for k, v in timing.items()})
            info["thresholds"] = thr
            ARTIFACTS.append(info)
        except Exception:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "FAILURE.txt").write_text(traceback.format_exc())
            raise
        print(f"  epochs {info['epochs_run']} (best {info['best_epoch']}) | calibration F1 "
              f"FP32 {evaluate(CAL_Y, cal['fp32'], thr['fp32_f1'])['f1']:.4f} / "
              f"INT8 {evaluate(CAL_Y, cal['int8'], thr['int8_f1'])['f1']:.4f} | "
              f"files {info['fp32_bytes']:,} / {info['int8_bytes']:,} bytes")
        save_progress()

# ---- Internal protocol lock: hash everything BEFORE any confirmation block is scored ----
(OUT / "lock/esa_thresholds.json").write_text(json.dumps(LOCKED_THRESHOLDS, indent=2))
locked = sorted(p for p in (OUT / "models/esa").rglob("*") if p.suffix in (".tflite", ".npz", ".keras"))
locked += [OUT / "lock/esa_thresholds.json", OUT / "data/esa_scaler.json", OUT / "data/esa_blocks.csv.gz",
           OUT / "data/esa_block_events.csv.gz", OUT / "data/esa_member_manifest.json"]
LOCK = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "statement": "Internal protocol lock (not an external preregistration). All models, thresholds, "
                     "preprocessing artifacts, labels, baseline parameters and evaluator code are fixed below; "
                     "every declared run, condition and target is evaluated once, without revision.",
        "confirmation_interval": "2001-10-01 (included) to 2002-10-01 (excluded)",
        "targets": [ESA_PRIMARY, ESA_SECONDARY], "conditions": list(CONDITION_NAMES), "seeds": SEEDS,
        "files": {str(p.relative_to(OUT)): sha256_file(p) for p in locked},
        "evaluator_code_sha256": code_fingerprint(EVALUATOR_OBJS)}
LOCK_PATH = OUT / "lock/esa_lock.json"
LOCK_PATH.write_text(json.dumps(LOCK, indent=2))
print(f"\nLOCKED: {len(LOCK['files'])} files + evaluator code hashed. Lock SHA-256: {sha256_file(LOCK_PATH)}")
print("No confirmation block has been scored yet.")
