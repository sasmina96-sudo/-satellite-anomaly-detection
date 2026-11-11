# STEP 13 — COST: bits, transmit time, transmit energy and ground-station minutes saved
BITS_PER_READING = 16                     # ASSUMED raw sample size (headers, timestamps, compression not modelled)
SUMMARY_BITS = 3 * 16 + 1                 # min, mean, max (16-bit each) + 1-bit alarm flag per block
TX_DC_POWER_W = 13.0                      # ISIS TXS S-band transmitter datasheet: 13 W typical DC, TX on, 33 dBm
INFO_RATES_BPS = {"4.30 Mbit/s (datasheet maximum)": 4.3e6,
                  "0.54 Mbit/s (lowest symbol rate, scaled)": 4.3e6 * 0.625 / 5}
GS_USD_PER_MINUTE = None                  # e.g. current AWS Ground Station rate from calculator.aws (leave None if unknown)
SCALE_CHANNELS = 1000                     # illustration: linear extrapolation to 1,000 similar channels

raw, pos, n_blocks = PRED["raw_readings"].to_numpy(), PRED["primary"].to_numpy(), len(PRED)
days = (T_END - T_CAL_END) / (86400 * 10**9)
BITS_P0 = int(raw.sum()) * BITS_PER_READING
cat = CONF_EVENTS["category"].str.lower()
RARE_IDS = CONF_EVENTS.loc[cat.str.contains("rare"), "event_id"].unique()
ANOM_IDS = CONF_EVENTS.loc[cat.eq("anomaly"), "event_id"].unique()


def policy_stats(flag):
    flag = np.asarray(flag, dtype=bool)
    raw_bits = int(raw[flag].sum()) * BITS_PER_READING
    p1 = n_blocks * SUMMARY_BITS + raw_bits
    caught = CONF_EVENTS.assign(f=flag[CONF_EVENTS["pos"].to_numpy()]).groupby("event_id")["f"].any()
    return {"Flagged blocks": int(flag.sum()),
            "Block recall": float((flag & pos).sum() / max(pos.sum(), 1)),
            "Rare-event recall (by ID)": float(caught.reindex(RARE_IDS).fillna(False).mean()) if len(RARE_IDS) else np.nan,
            "True anomalies caught": f"{int(caught.reindex(ANOM_IDS).fillna(False).sum())}/{len(ANOM_IDS)}",
            "False-alarm blocks": int((flag & ~pos).sum()),
            "P1 bits": p1, "P1 saving %": 100 * (1 - p1 / BITS_P0),
            "P2 bits": raw_bits, "P2 saving %": 100 * (1 - raw_bits / BITS_P0),
            "False alarms in sent raw %": 100 * raw[flag & ~pos].sum() / max(raw[flag].sum(), 1)}


per_run = []
for arch in ARCHS:
    for seed in SEEDS:
        for fmt in ("fp32", "int8"):
            per_run.append({"Detector": f"{NAME[arch]} {fmt.upper()}", "Seed": seed,
                            **policy_stats(PRED[f"{arch}_seed{seed}_{fmt}_flag"])})
for name in Baselines.NAMES:
    per_run.append({"Detector": f"{name} (control)", "Seed": np.nan, **policy_stats(PRED[f"{name}_flag"])})
per_run.append({"Detector": "Oracle (true labels)", "Seed": np.nan, **policy_stats(pos)})
DOWNLINK_RUNS = pd.DataFrame(per_run)
DOWNLINK_RUNS.to_csv(OUT / "results/downlink_policy_per_run.csv", index=False)
numeric = DOWNLINK_RUNS.select_dtypes("number").columns.drop("Seed")
DOWNLINK = DOWNLINK_RUNS.groupby("Detector", sort=False)[list(numeric)].mean()
DOWNLINK["True anomalies caught"] = DOWNLINK_RUNS.groupby("Detector", sort=False)["True anomalies caught"].agg(lambda v: " · ".join(v))
DOWNLINK.to_csv(OUT / "results/downlink_policy_summary.csv")

print(f"Confirmation year: {n_blocks:,} channel-blocks, {raw.sum():,} native readings")
print(f"P0 (send everything): {BITS_P0 / 8e6:.2f} MB per year for {len(ESA_CHANNELS)} channels\n")
show = ["Flagged blocks", "Block recall", "Rare-event recall (by ID)", "True anomalies caught",
        "P1 saving %", "P2 saving %", "False alarms in sent raw %"]
print(DOWNLINK[show].round(3).to_string())

# ---- convert P1 savings into transmit time, transmit energy and ground-station minutes (per year) ----
compute_j_year = POWER.set_index(["Model", "Format"])["Estimate total (nJ)"] * INFERENCES_PER_DAY * 365 * 1e-9
cost_rows = []
for detector, row in DOWNLINK.iterrows():
    saved_bits = (BITS_P0 - row["P1 bits"]) * 365 / days
    for rate_name, rate in INFO_RATES_BPS.items():
        seconds = saved_bits / rate
        entry = {"Detector": detector, "Link": rate_name, "Bits saved / year (Mbit)": saved_bits / 1e6,
                 "Transmit time saved / year (s)": seconds, "Transmit energy saved / year (J)": seconds * TX_DC_POWER_W,
                 "Ground-station minutes saved / year": seconds / 60,
                 f"Minutes saved / year at {SCALE_CHANNELS:,} channels": seconds / 60 * SCALE_CHANNELS / len(ESA_CHANNELS)}
        if GS_USD_PER_MINUTE:
            entry["USD saved / year"] = seconds / 60 * GS_USD_PER_MINUTE
        cost_rows.append(entry)
COST = pd.DataFrame(cost_rows)
COST.to_csv(OUT / "results/downlink_cost_per_year.csv", index=False)
print("\nPer year, policy P1:")
print(COST.round(3).to_string(index=False))

print("\nEnergy balance (estimates): detector compute energy per year vs transmit energy saved per year")
for arch in ARCHS:
    for fmt in ("FP32", "INT8"):
        e_compute = compute_j_year[(NAME[arch], fmt)]
        e_saved = COST[(COST["Detector"] == f"{NAME[arch]} {fmt}")]["Transmit energy saved / year (J)"]
        print(f"  {NAME[arch]:5} {fmt}: compute ≈ {e_compute * 1e3:.3f} mJ/year | transmit saved ≈ "
              f"{e_saved.min():.1f}–{e_saved.max():.1f} J/year (link-rate range)")

# ---- Fig. 2: what each detector saves (left) next to what it still catches (right) ----
from matplotlib.patches import Patch
order = DOWNLINK.index[::-1]
colors = [COLORS["fp32"] if "FP32" in d else COLORS["int8"] if "INT8" in d else COLORS["control"] for d in order]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
panels = [("P1 saving %", "Downlink saved vs send-everything (%)", 100, "{:.1f}%"),
          ("Rare-event recall (by ID)", "Rare events with ≥1 flagged block", 1, "{:.2f}")]
for ax, (column, label, limit, fmt_) in zip(axes, panels):
    values = DOWNLINK.loc[order, column].to_numpy(dtype=float)
    ax.barh(range(len(order)), values, height=0.6, color=colors, edgecolor="white", linewidth=1)
    for y_pos, value in enumerate(values):
        ax.text(value + limit * 0.01, y_pos, fmt_.format(value), va="center", fontsize=7.5, color="#3a3936")
    ax.set_xlim(0, limit * 1.15)
    ax.set_xlabel(label)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order)
fig.legend(handles=[Patch(color=COLORS["fp32"], label="FP32 model"), Patch(color=COLORS["int8"], label="INT8 model"),
                    Patch(color=COLORS["control"], label="Control / oracle")], frameon=False, fontsize=8,
           loc="lower center", ncol=3)
fig.suptitle("Policy P1 on the ESA confirmation year: summary for every block, raw data only for flagged blocks",
             fontsize=9.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "figures/fig2_downlink_vs_recall.png", dpi=300)
fig.savefig(OUT / "figures/fig2_downlink_vs_recall.pdf")
plt.show()
