# EXTRA ANALYSIS — (A) flag-budget operating curve with a random-flag baseline, (B) lossless-compression baseline
# Reads the finished run in MyDrive/Satellite_Full_Paper_Run; writes results to <run>/extra/. Nothing is retrained.
import os, re, json, lzma, zlib, time
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/Satellite_Full_Paper_Run")
except ImportError:
    BASE = Path(os.environ.get("SAT_BASE", str(Path.cwd() / "Satellite_Full_Paper_Run")))

runs = sorted(p.parent.parent for p in (BASE / "runs").glob("*/results/esa_confirmation_predictions.csv.gz"))
assert runs, f"No finished run found under {BASE / 'runs'}"
RUN = runs[-1]
OUT = RUN / "extra"
OUT.mkdir(exist_ok=True)
print("Using run:", RUN.name)

# ---------------- load confirmation predictions and event membership ----------------
pred = pd.read_csv(RUN / "results/esa_confirmation_predictions.csv.gz")
blocks = pd.read_csv(RUN / "data/esa_blocks.csv.gz", usecols=["channel", "block"])
blocks["row"] = np.arange(len(blocks))
events = pd.read_csv(RUN / "data/esa_block_events.csv.gz")
key = pred[["channel", "block"]].merge(blocks, on=["channel", "block"], how="left", validate="one_to_one")
assert key["row"].notna().all()
pos = pd.Series(np.arange(len(pred)), index=key["row"].to_numpy())
ev = events[events["row"].isin(pos.index)].copy()
ev["pos"] = pos.loc[ev["row"]].to_numpy()
cat = ev["category"].astype(str).str.lower()
RARE_IDS = ev.loc[cat.str.contains("rare"), "event_id"].unique()
ANOM_IDS = ev.loc[cat.eq("anomaly"), "event_id"].unique()
EV_POS, EV_ID = ev["pos"].to_numpy(), ev["event_id"].to_numpy()
raw = pred["raw_readings"].to_numpy()
y = pred["primary"].to_numpy().astype(bool)
K = len(pred)
B0 = int(raw.sum()) * 16
SUMMARY_BITS = 49
print(f"{K:,} confirmation blocks | {raw.sum():,} native readings | {len(RARE_IDS)} rare events | {len(ANOM_IDS)} true anomalies")

NAME = {"lstm": "LSTM", "dense": "Dense"}
DETECTORS = {}
for c in pred.columns:
    m = re.fullmatch(r"(lstm|dense)_seed(\d+)_(fp32|int8)_score", c)
    if m:
        DETECTORS[c] = (f"{NAME[m[1]]} {m[3].upper()}", int(m[2]), c.replace("_score", "_flag"))
for b in ("PCA4", "ZScore", "Persistence"):
    if f"{b}_score" in pred:
        DETECTORS[f"{b}_score"] = (b, None, f"{b}_flag")


def stats(flag):
    flag = np.asarray(flag, dtype=bool)
    raw_bits = int(raw[flag].sum()) * 16
    caught = pd.Series(flag[EV_POS]).groupby(EV_ID).any()
    return {"flagged": int(flag.sum()), "flag_pct": 100 * flag.mean(),
            "precision": (flag & y).sum() / max(flag.sum(), 1), "block_recall": (flag & y).sum() / max(y.sum(), 1),
            "rare_event_recall": float(caught.reindex(RARE_IDS).fillna(False).mean()) if len(RARE_IDS) else np.nan,
            "anomaly_caught": int(caught.reindex(ANOM_IDS).fillna(False).sum()),
            "p1_saving_pct": 100 * (1 - (SUMMARY_BITS * K + raw_bits) / B0),
            "p2_saving_pct": 100 * (1 - raw_bits / B0)}


def top_budget(score, pct):
    n = int(round(pct / 100 * K))
    s = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=-np.inf)
    flag = np.zeros(K, dtype=bool)
    flag[np.argsort(-s, kind="stable")[:n]] = True
    return flag


def random_mean(n_flag, draws=50, seed=0):
    """Mean statistics of `draws` sets of n_flag blocks chosen uniformly at random (no detector)."""
    g = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        flag = np.zeros(K, dtype=bool)
        flag[g.choice(K, size=n_flag, replace=False)] = True
        out.append(stats(flag))
    return {k: float(np.mean([d[k] for d in out])) for k in out[0]}


# ---------------- (A) flag-budget sweep ----------------
BUDGETS = [0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 30, 50]
rows = []
t0 = time.time()
for col, (name, seed, flag_col) in DETECTORS.items():
    s = pred[col].to_numpy()
    locked = pred[flag_col].to_numpy().astype(bool)
    rnd = random_mean(int(locked.sum()))           # random flags with exactly the same number of flagged blocks
    rows.append({"detector": name, "seed": seed, "budget_pct": "locked", **stats(locked),
                 "random_same_budget_rare_event_recall": rnd["rare_event_recall"],
                 "random_same_budget_block_recall": rnd["block_recall"]})
    for b in BUDGETS:
        rows.append({"detector": name, "seed": seed, "budget_pct": b, **stats(top_budget(s, b))})
for b in BUDGETS:
    rows.append({"detector": "Random flags", "seed": None, "budget_pct": b,
                 **random_mean(int(round(b / 100 * K)), seed=int(b * 100))})
SWEEP = pd.DataFrame(rows)
print(f"(A) sweep done in {time.time() - t0:.0f} s")
SWEEP.to_csv(OUT / "sweep_per_run.csv", index=False)
num_cols = ["flagged", "flag_pct", "precision", "block_recall", "rare_event_recall", "anomaly_caught",
            "p1_saving_pct", "p2_saving_pct", "random_same_budget_rare_event_recall", "random_same_budget_block_recall"]
SUMMARY = SWEEP.groupby(["detector", "budget_pct"], sort=False)[num_cols].mean().reset_index()
SUMMARY.to_csv(OUT / "sweep_summary.csv", index=False)

# ---------------- (B) lossless-compression baseline ----------------
CACHE = BASE / "data_cache" / "esa_mission2"
NS = 10**9
T_START = int(np.datetime64("2000-01-01", "ns").astype(np.int64))
T_CONF = int(np.datetime64("2001-10-01", "ns").astype(np.int64))
T_END = int(np.datetime64("2002-10-01", "ns").astype(np.int64))
BLOCK_NS = 600 * 8 * NS


def load_channel(path):
    try:
        obj = pd.read_pickle(path, compression="zip")
    except Exception:
        obj = pd.read_pickle(path)
    series = obj if isinstance(obj, pd.Series) else obj.iloc[:, 0]
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    t = index.to_numpy(dtype="datetime64[ns]").view(np.int64)
    v = pd.to_numeric(pd.Series(series.to_numpy()), errors="coerce").to_numpy(dtype=np.float64)
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def encode(q, delta):
    if delta and len(q):
        q = (np.diff(q.astype(np.int64), prepend=0) & 0xFFFF)
    return q.astype("<u2").tobytes()


LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 1 << 22}]   # preset 9, 4 MiB dictionary
COMPRESS = {"zlib-9": lambda b: len(zlib.compress(b, 9)) if b else 0,
            "LZMA2-9": lambda b: len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)) if b else 0}
METHODS = [(c, d) for c in COMPRESS for d in (False, True)]

chan = {}
total_readings = 0
for ch in sorted(pred["channel"].unique(), key=lambda c: int(c.split("_")[1])):
    t, v = load_channel(CACHE / f"{ch}.zip")
    m = (t >= T_CONF) & (t < T_END)
    t, v = t[m], v[m]
    finite = np.isfinite(v)
    lo, hi = (v[finite].min(), v[finite].max()) if finite.any() else (0.0, 0.0)
    q = np.zeros(len(v), dtype=np.uint16)                     # 16-bit readings, as in the bit model; 0 = missing
    q[finite] = 1 if hi == lo else np.rint((v[finite] - lo) / (hi - lo) * 65534).astype(np.int64) + 1
    day = (t - T_CONF) // (86400 * NS)
    blk = ((t - T_START) // BLOCK_NS).astype(np.int32)
    cuts = np.searchsorted(day, np.arange(1, 366))
    chan[ch] = {"q": q, "blk": blk, "cuts": cuts}
    total_readings += len(q)
print(f"Native readings re-read from the channel files: {total_readings:,} (prediction table: {raw.sum():,})")


def compressed_bits(mask_by_channel, method, daily=True):
    comp, delta = COMPRESS[method[0]], method[1]
    total = 0
    for ch, d in chan.items():
        keep = mask_by_channel(ch, d)
        if not daily:                                   # whole confirmation year per channel (best case)
            total += comp(encode(d["q"][keep], delta))
            continue
        for part_q, part_k in zip(np.split(d["q"], d["cuts"]), np.split(keep, d["cuts"])):   # one chunk per channel-day
            total += comp(encode(part_q[part_k], delta))
    return 8 * total


ALL = lambda ch, d: np.ones(len(d["q"]), dtype=bool)
comp_rows = []
t0 = time.time()
for method in METHODS:
    bits = compressed_bits(ALL, method)
    year_bits = compressed_bits(ALL, method, daily=False)
    comp_rows.append({"method": f"{method[0]}{' + delta' if method[1] else ''}", "P0_raw_Mbit": B0 / 1e6,
                      "P0_compressed_Mbit": bits / 1e6, "ratio": B0 / bits,
                      "P0_compressed_whole_year_Mbit": year_bits / 1e6, "ratio_whole_year": B0 / year_bits})
    print(f"  {comp_rows[-1]['method']:<16} daily chunks {bits / 1e6:8.2f} Mbit (x{B0 / bits:.2f}) | "
          f"whole year {year_bits / 1e6:8.2f} Mbit (x{B0 / year_bits:.2f})   [{time.time() - t0:.0f} s]")
COMP_P0 = pd.DataFrame(comp_rows)
best = METHODS[int(COMP_P0["P0_compressed_Mbit"].idxmin())]
best_name = COMP_P0.loc[COMP_P0["P0_compressed_Mbit"].idxmin(), "method"]
B0c = COMP_P0["P0_compressed_Mbit"].min() * 1e6
print(f"Best lossless method: {best_name} -> P0 {B0 / 1e6:.1f} Mbit raw, {B0c / 1e6:.2f} Mbit compressed")

flag_sets = {f"{n}|{s}": pred[fc].to_numpy().astype(bool) for _, (n, s, fc) in DETECTORS.items()}
flag_sets["Oracle (true labels)|None"] = y
p1_rows = []
for key_, flag in flag_sets.items():
    name, seed = key_.split("|")
    flagged = {(ch, int(b)) for ch, b in zip(pred["channel"][flag], pred["block"][flag])}
    by_ch = {}
    for ch in chan:
        by_ch[ch] = np.array(sorted(b for c, b in flagged if c == ch), dtype=np.int64)
    bits = compressed_bits(lambda ch, d: np.isin(d["blk"], by_ch[ch]), best)
    p1c = SUMMARY_BITS * K + bits
    p1_rows.append({"detector": name, "seed": seed, "P1_compressed_Mbit": p1c / 1e6,
                    "saving_vs_P0_compressed_pct": 100 * (1 - p1c / B0c),
                    "saving_vs_P0_raw_pct": 100 * (1 - p1c / B0)})
B0c_year = COMP_P0["P0_compressed_whole_year_Mbit"].min() * 1e6
for r in p1_rows:
    r["saving_vs_P0_compressed_whole_year_pct"] = 100 * (1 - r["P1_compressed_Mbit"] * 1e6 / B0c_year)
COMP_P1 = pd.DataFrame(p1_rows).groupby("detector", sort=False)[
    ["P1_compressed_Mbit", "saving_vs_P0_compressed_pct", "saving_vs_P0_compressed_whole_year_pct",
     "saving_vs_P0_raw_pct"]].mean().reset_index()
COMP_P0.to_csv(OUT / "compression_P0.csv", index=False)
COMP_P1.to_csv(OUT / "compression_P1.csv", index=False)

# ---------------- write EXTRA_NUMBERS.md ----------------
def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


LOCKED = SUMMARY[SUMMARY["budget_pct"].astype(str) == "locked"][
    ["detector", "flagged", "flag_pct", "block_recall", "rare_event_recall", "random_same_budget_rare_event_recall",
     "random_same_budget_block_recall", "anomaly_caught", "p1_saving_pct"]]
CURVE = SUMMARY[SUMMARY["budget_pct"].astype(str) != "locked"].assign(budget=lambda d: d["budget_pct"].astype(float))
pivot = lambda col: CURVE.pivot_table(index="detector", columns="budget", values=col, sort=False).reset_index()
pivot_md = lambda col: md(pivot(col).rename(columns=lambda c: c if c == "detector" else f"{c:g}%"))
lines = [f"# Extra analysis for run {RUN.name}", "",
         f"Confirmation blocks: {K}; native readings: {int(raw.sum())} (re-read from channel files: {total_readings}); "
         f"rare events: {len(RARE_IDS)}; true anomalies: {len(ANOM_IDS)}", "",
         "## A1. Locked thresholds vs random flags with the same number of flagged blocks (means over seeds; 50 random draws)",
         "", md(LOCKED), "",
         "## A2. Rare-event recall (by ID) when the top-x% highest-scoring blocks are flagged", "", pivot_md("rare_event_recall"), "",
         "## A3. Block recall at the same budgets", "", pivot_md("block_recall"), "",
         "## A4. P1 saving % at the same budgets", "", pivot_md("p1_saving_pct"), "",
         "## B1. Lossless compression of P0 (16-bit readings; one chunk per channel-day, and whole year per channel)", "",
         md(COMP_P0), "", f"Best method (daily chunks), used for P1: {best_name}", "",
         "## B2. P1 with its raw part compressed by the best method (summaries left uncompressed)", "", md(COMP_P1), ""]
(OUT / "EXTRA_NUMBERS.md").write_text("\n".join(lines), encoding="utf-8")

# ---------------- Fig. 3: event capture vs flag budget ----------------
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
STYLE = {"LSTM INT8": ("#eb6834", "o", "-"), "Dense INT8": ("#eb6834", "s", "--"),
         "ZScore": ("#1baf7a", "D", "-"), "PCA4": ("#6250d6", "^", "-"), "Random flags": ("#8a8984", None, ":")}
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for name, (color, marker, ls) in STYLE.items():
    d = SUMMARY[(SUMMARY["detector"] == name) & (SUMMARY["budget_pct"].astype(str) != "locked")]
    ax.plot(d["budget_pct"].astype(float), d["rare_event_recall"], color=color, marker=marker, ls=ls, lw=1.2,
            ms=3.2, label=name, markeredgecolor="white", markeredgewidth=0.4)
for name in ("LSTM INT8", "Dense INT8"):
    d = SUMMARY[(SUMMARY["detector"] == name) & (SUMMARY["budget_pct"].astype(str) == "locked")]
    ax.scatter(d["flag_pct"], d["rare_event_recall"], s=34, facecolor="none", edgecolor="#2b2a27", lw=0.8, zorder=5,
               marker=STYLE[name][1])
ax.set_xscale("log")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xticks(BUDGETS)
ax.set_xticklabels([f"{b:g}" for b in BUDGETS])
ax.set_xlabel("Blocks flagged (% of confirmation year)")
ax.set_ylabel("Rare events with a flagged block")
ax.set_ylim(0, 1.02)
ax.grid(color="#e4e3df", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.3, loc="lower right")
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig3_event_capture.pdf")
fig.savefig(OUT / "fig3_event_capture.png", dpi=200)
plt.show()
print("\n".join(lines))
print(f"\nSaved to {OUT}")
