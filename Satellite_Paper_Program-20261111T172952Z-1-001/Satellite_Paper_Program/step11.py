# STEP 11 — Tables 1–3 (+ extra tables) and Fig. 1
import matplotlib.pyplot as plt

R, A = pd.DataFrame(ROWS), pd.DataFrame(ARTIFACTS)
R.to_csv(OUT / "results/all_metrics.csv", index=False)
A.to_csv(OUT / "results/artifacts.csv", index=False)
PRIMARY = {"OPS-SAT validation": "annotated anomaly", "ESA confirmation": ESA_PRIMARY}


def by_seed(cohort, model, cond, target=None, metric="f1"):
    q = R[(R["cohort"] == cohort) & (R["target"] == (target or PRIMARY[cohort])) & (R["model"] == model)
          & (R["condition"] == cond)]
    return q.set_index("seed")[metric].sort_index()


def ms(s):
    return f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


TABLES = {}
TABLES["Table 1 — Mean F1 ± sample SD over seeds (primary threshold rule)"] = pd.DataFrame([
    {"Cohort": c, "Model": NAME[a], "FP32 F1": ms(by_seed(c, a, "fp32_f1")), "INT8 F1": ms(by_seed(c, a, "int8_f1")),
     "Mean change": f"{(by_seed(c, a, 'int8_f1') - by_seed(c, a, 'fp32_f1')).mean():+.4f}",
     "SD of change": f"{(by_seed(c, a, 'int8_f1') - by_seed(c, a, 'fp32_f1')).std(ddof=1):.4f}"}
    for c in PRIMARY for a in ARCHS])
TABLES["Table 2 — Every training seed (primary threshold rule)"] = pd.DataFrame([
    {"Cohort": c, "Model": NAME[a], "Seed": int(s), "FP32 F1": f"{by_seed(c, a, 'fp32_f1')[s]:.4f}",
     "INT8 F1": f"{by_seed(c, a, 'int8_f1')[s]:.4f}",
     "Change": f"{by_seed(c, a, 'int8_f1')[s] - by_seed(c, a, 'fp32_f1')[s]:+.4f}"}
    for c in PRIMARY for a in ARCHS for s in SEEDS])
TABLES["Table 3 — Fixed controls, separately calibrated max-F1 thresholds"] = pd.DataFrame([
    {"Cohort": c, "Baseline": b, "F1": f"{by_seed(c, b, 'f1').iloc[0]:.4f}",
     "F1 (nominal P99)": f"{by_seed(c, b, 'p99').iloc[0]:.4f}"} for c in PRIMARY for b in Baselines.NAMES])
TABLES["Table 4 — Mean F1 under every declared operating condition"] = pd.DataFrame([
    {"Cohort": c, "Model": NAME[a], **{CONDITION_NAMES[k]: f"{by_seed(c, a, k).mean():.4f}"
                                       for k in ("fp32_f1", "int8_f1", "int8_transfer", "fp32_p99", "int8_p99")}}
    for c in PRIMARY for a in ARCHS])
TABLES["Table 5 — ESA confirmation: precision, recall and ranking (mean over seeds, primary target)"] = pd.DataFrame([
    {"Model": NAME[a], "Format": fmt.upper(),
     **{m.replace("_", " ").title(): f"{by_seed('ESA confirmation', a, fmt + '_f1', metric=m).mean():.4f}"
        for m in ("precision", "recall", "f1", "roc_auc", "average_precision")}}
    for a in ARCHS for fmt in ("fp32", "int8")])
TABLES["Table 6 — ESA secondary target (anomaly only; rare-only blocks excluded) — descriptive"] = pd.DataFrame([
    {"Model": NAME.get(m, m), "Condition": CONDITION_NAMES[k],
     "Mean F1": f"{by_seed('ESA confirmation', m, k, target=ESA_SECONDARY).mean():.4f}",
     "Mean recall": f"{by_seed('ESA confirmation', m, k, target=ESA_SECONDARY, metric='recall').mean():.4f}"}
    for m, k in [(a, c) for a in ARCHS for c in ("fp32_f1", "int8_f1")] + [(b, "f1") for b in Baselines.NAMES]])
art_rows = []
for (cohort, arch), g in A.groupby(["cohort", "arch"], sort=False):
    art_rows.append({"Cohort": cohort, "Model": NAME[arch],
                     "FP32 file (bytes)": ", ".join(f"{v:,}" for v in sorted(g["fp32_bytes"].unique())),
                     "INT8 file (bytes)": ", ".join(f"{v:,}" for v in sorted(g["int8_bytes"].unique())),
                     "INT8 floating tensors": int(g["int8_float_tensors"].max()),
                     "Default-quantizer floating tensors (diagnostic)": ", ".join(map(str, g["default_quantizer_float_tensors"])),
                     "Median latency FP32 / INT8 (µs)": f"{g['fp32_median_us'].mean():.1f} / {g['int8_median_us'].mean():.1f}",
                     "INT8 input clipping": f"{100 * g['int8_clip_fraction'].mean():.3f}%"})
TABLES["Table 7 — Exported files, tensor audits, CPU latency (batch 1, 1 thread) and clipping"] = pd.DataFrame(art_rows)
TABLES["ESA cohort description"] = pd.DataFrame([{"Quantity": k, "Value": f"{v:,}"} for k, v in ESA_COHORT.items()])

for title, table in TABLES.items():
    print(f"\n{title}\n{table.to_string(index=False)}")
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower().split("—")[0]).strip("_")
    table.to_csv(OUT / f"results/{slug}.csv", index=False)

# ---- Fig. 1: paired precision formats (dots = seeds, square ± bar = mean ± sample SD) ----
COLORS = {"fp32": "#2a78d6", "int8": "#eb6834", "control": "#1baf7a"}   # validated categorical slots 1–3
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
for ax, cohort in zip(axes, PRIMARY):
    ticks, ticklabels = [], []
    for a_i, arch in enumerate(ARCHS):
        fp, i8 = by_seed(cohort, arch, "fp32_f1"), by_seed(cohort, arch, "int8_f1")
        x0, x1 = a_i * 2.6, a_i * 2.6 + 1
        for s in fp.index:
            ax.plot([x0, x1], [fp[s], i8[s]], color="#b8b7b2", lw=1, zorder=1)
        for x, series, fmt in ((x0, fp, "fp32"), (x1, i8, "int8")):
            ax.scatter([x] * len(series), series.values, s=28, color=COLORS[fmt], edgecolor="white",
                       linewidth=1, zorder=3, label=fmt.upper() if a_i == 0 else None)
            ax.errorbar(x + 0.22, series.mean(), yerr=series.std(ddof=1), fmt="s", ms=4, color=COLORS[fmt],
                        capsize=3, lw=1.5, zorder=2)
        ticks += [x0, x1]
        ticklabels += [f"{NAME[arch]}\nFP32", f"{NAME[arch]}\nINT8"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_title(cohort, fontsize=10)
    ax.set_ylabel("F1")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "figures/fig1_paired_f1.png", dpi=300)
fig.savefig(OUT / "figures/fig1_paired_f1.pdf")
plt.show()
