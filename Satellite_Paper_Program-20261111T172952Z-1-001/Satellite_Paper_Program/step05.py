# STEP 5 — OPS-SAT: 2 architectures × 3 seeds → FP32 + INT8 exports → 5 operating conditions
for arch in ARCHS:
    for seed in SEEDS:
        run_dir = OUT / "models/opssat" / f"{arch}_seed{seed}"
        print(f"\n=== OPS-SAT | {NAME[arch]} | seed {seed} ===")
        try:
            files, info = train_and_export(arch, seed, "OPS-SAT", OPS_X_FIT, OPS_X_VAL_NOM, OPS_REP,
                                           OPS_X_CHECK, run_dir)
            scores = {}
            for fmt in ("fp32", "int8"):
                recon, clipped = Runner(files[fmt]).run(OPS_X_VAL)
                y, scores[fmt] = ops_segment_scores(window_mse(OPS_X_VAL, recon))
                info[f"{fmt}_clip_fraction"] = float(clipped.sum() / OPS_X_VAL.size)
            thr = threshold_set(y, scores)
            for cond, (fmt, t) in conditions(thr).items():
                ROWS.append(dict(cohort="OPS-SAT validation", target="annotated anomaly", model=arch,
                                 kind="learned", seed=seed, condition=cond, **evaluate(y, scores[fmt], t)))
            for fmt, timing in latency_benchmark(files, OPS_X_BENCH).items():
                info.update({f"{fmt}_{k}": float(v) for k, v in timing.items()})
            info["thresholds"] = thr
            pd.DataFrame({"label": y, "fp32_score": scores["fp32"], "int8_score": scores["int8"]}).to_csv(
                run_dir / "validation_segment_scores.csv", index=False)
            ARTIFACTS.append(info)
        except Exception:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "FAILURE.txt").write_text(traceback.format_exc())
            raise
        f1 = {r["condition"]: r["f1"] for r in ROWS[-5:]}
        print(f"  epochs {info['epochs_run']} (best {info['best_epoch']}) | F1 FP32 {f1['fp32_f1']:.4f} → "
              f"INT8 {f1['int8_f1']:.4f} | files {info['fp32_bytes']:,} / {info['int8_bytes']:,} bytes | "
              f"median latency {info['fp32_median_us']:.1f} / {info['int8_median_us']:.1f} µs | "
              f"default-quantizer float tensors: {info['default_quantizer_float_tensors']}")
        save_progress()
print("\nOPS-SAT complete.")
