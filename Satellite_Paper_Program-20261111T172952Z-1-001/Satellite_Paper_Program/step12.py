# STEP 12 — POWER: energy per inference (ESTIMATE) and per day for the ESA monitoring scheme
E_PJ = {"fp32_mult": 3.7, "fp32_add": 0.9,        # Horowitz 2014, 45 nm, 0.9 V (picojoules)
        "int8_mult": 0.2, "int32_add": 0.1,
        "sram_32kb_read_64bit": 20.0}
P_ACTIVE_W = 1.0                                   # ASSUMED extra CPU power during one inference (edit)
INFERENCES_PER_DAY = len(ESA_CHANNELS) * (24 * 3600 // (SAMPLE_NS // 10**9 * BLOCK_LEN))   # 11 × 18 = 198
ANALYTIC_MACS = {"lstm": 4 * (1 + 16) * 16 * WINDOW + 4 * (16 + 16) * 16 * WINDOW + 16 * WINDOW,
                 "dense": 8 * 16 + 16 * 4 + 4 * 16 + 16 * 8}
power_rows = []
for arch in ARCHS:
    lat = A[(A["cohort"] == "ESA") & (A["arch"] == arch)]
    for fmt in ("fp32", "int8"):
        model_bytes = (OUT / "models/esa" / f"{arch}_seed{SEEDS[0]}" / f"model_{fmt}.tflite").read_bytes()
        macs, ops = count_macs(model_bytes)
        macs = macs or ANALYTIC_MACS[arch]
        per_mac = E_PJ["fp32_mult"] + E_PJ["fp32_add"] if fmt == "fp32" else E_PJ["int8_mult"] + E_PJ["int32_add"]
        weight_bytes = 4 if fmt == "fp32" else 1
        arith_nj = macs * per_mac / 1000
        fetch_nj = macs * weight_bytes / 8 * E_PJ["sram_32kb_read_64bit"] / 1000
        median_us = float(lat[f"{fmt}_median_us"].mean())
        power_rows.append({"Model": NAME[arch], "Format": fmt.upper(), "MACs (graph)": macs,
                           "MACs (analytic)": ANALYTIC_MACS[arch], "Arithmetic (nJ)": arith_nj,
                           "Weight fetch (nJ)": fetch_nj, "Estimate total (nJ)": arith_nj + fetch_nj,
                           "Per day, 198 inferences (µJ)": (arith_nj + fetch_nj) * INFERENCES_PER_DAY / 1000,
                           "Median CPU latency (µs)": median_us,
                           f"Latency × {P_ACTIVE_W:g} W (µJ)": median_us * P_ACTIVE_W,
                           "Graph ops": ", ".join(f"{k}×{v}" for k, v in sorted(ops.items()))})
POWER = pd.DataFrame(power_rows)
POWER.to_csv(OUT / "results/power_estimate.csv", index=False)
print(POWER.drop(columns=["Graph ops"]).round(2).to_string(index=False))
for arch in ARCHS:
    p = POWER[POWER["Model"] == NAME[arch]].set_index("Format")
    print(f"\n{NAME[arch]}: estimated arithmetic+fetch energy FP32/INT8 = "
          f"{p.loc['FP32', 'Estimate total (nJ)'] / p.loc['INT8', 'Estimate total (nJ)']:.1f}× | "
          f"measured CPU latency FP32/INT8 = {p.loc['FP32', 'Median CPU latency (µs)'] / p.loc['INT8', 'Median CPU latency (µs)']:.2f}×")
print("\nGraph op counts (LSTM INT8):", POWER.iloc[1]["Graph ops"])
print("\nReading: the estimate isolates arithmetic + weight traffic only. If the latency ratio is far smaller than")
print("the arithmetic ratio, interpreter/control overhead dominates on this CPU — report both, measured vs estimated.")
