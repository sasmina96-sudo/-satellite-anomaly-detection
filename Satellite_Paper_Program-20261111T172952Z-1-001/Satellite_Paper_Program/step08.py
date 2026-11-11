# STEP 8 — ESA: 600 s carry-forward sampling → 80-minute channel-blocks → labels, splits, scaling
SAMPLE_NS = 600 * 10**9
BLOCK_LEN = 8
BLOCK_NS = SAMPLE_NS * BLOCK_LEN
ESA_PRIMARY, ESA_SECONDARY = "anomaly or rare event (primary)", "anomaly only (secondary)"


def ns(day):
    return int(np.datetime64(day, "ns").astype(np.int64))


T_START, T_FIT_END, T_CAL_END, T_END = ns("2000-01-01"), ns("2001-07-01"), ns("2001-10-01"), ns("2002-10-01")


def find_col(df, *candidates):
    lookup = {re.sub(r"[\s_]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[\s_]", "", cand.lower())
        if key in lookup:
            return lookup[key]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def to_ns(values):
    try:
        ts = pd.to_datetime(values, utc=True, format="mixed")
    except (TypeError, ValueError):
        ts = pd.to_datetime(values, utc=True)
    return ts.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").view(np.int64)


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
    order = np.argsort(t, kind="stable")            # stable: duplicates keep file order → last one wins
    return t[order], v[order]


labels = pd.read_csv(ESA_DIR / "labels.csv")
types = pd.read_csv(ESA_DIR / "anomaly_types.csv")
print("labels.csv columns:       ", list(labels.columns))
print("anomaly_types.csv columns:", list(types.columns))
category = types[find_col(types, "Category")].astype(str).str.strip()
category.index = types[find_col(types, "ID")].astype(str).str.strip()
category = category[~category.index.duplicated()]
ann = pd.DataFrame({"event_id": labels[find_col(labels, "ID")].astype(str).str.strip(),
                    "channel": labels[find_col(labels, "Channel")].astype(str).str.strip().str.lower(),
                    "start": to_ns(labels[find_col(labels, "StartTime", "Start")]),
                    "end": to_ns(labels[find_col(labels, "EndTime", "End")])})
ann["category"] = ann["event_id"].map(category).fillna("UNKNOWN")
ann["is_anomaly"] = ann["category"].str.lower().eq("anomaly")
ann["is_rare"] = ann["category"].str.lower().str.contains("rare")
print("Event categories (whole mission):", ann.drop_duplicates("event_id")["category"].value_counts().to_dict())
if ann["category"].eq("UNKNOWN").any():
    print("⚠ IDs missing from anomaly_types.csv:", ann.loc[ann["category"].eq("UNKNOWN"), "event_id"].unique()[:10])
ann = ann[ann["channel"].isin(ESA_CHANNELS)].reset_index(drop=True)

grid = np.arange(T_START, T_END, SAMPLE_NS, dtype=np.int64)
assert len(grid) % BLOCK_LEN == 0
BLOCK_START = grid[::BLOCK_LEN]
N_PER_CH = len(BLOCK_START)
frames, value_blocks, event_parts = [], [], []
for c_i, ch in enumerate(ESA_CHANNELS):
    t, v = load_channel(ESA_DIR / f"{ch}.zip")
    pos = np.searchsorted(t, grid, side="right") - 1        # latest observation at or before each grid time
    sampled = np.full(len(grid), np.nan)
    sampled[pos >= 0] = v[pos[pos >= 0]]                    # no backfilling before the first observation
    sampled = sampled.reshape(-1, BLOCK_LEN)
    raw_count = (np.searchsorted(t, BLOCK_START + BLOCK_NS, "left")
                 - np.searchsorted(t, BLOCK_START, "left"))  # native readings inside each block (for downlink)
    flags = {k: np.zeros(N_PER_CH, dtype=bool) for k in ("any_annotation", "primary", "anomaly", "rare")}
    for e in ann[ann["channel"].eq(ch)].itertuples(index=False):
        lo = np.searchsorted(BLOCK_START, e.start - BLOCK_NS, side="right")   # block end > event start
        hi = np.searchsorted(BLOCK_START, e.end, side="right")                # block start ≤ event end
        if hi <= lo:
            continue
        flags["any_annotation"][lo:hi] = True
        flags["primary"][lo:hi] |= bool(e.is_anomaly or e.is_rare)
        flags["anomaly"][lo:hi] |= bool(e.is_anomaly)
        flags["rare"][lo:hi] |= bool(e.is_rare)
        event_parts.append(pd.DataFrame({"row": c_i * N_PER_CH + np.arange(lo, hi),
                                         "event_id": e.event_id, "category": e.category}))
    frames.append(pd.DataFrame({"channel": ch, "block": np.arange(N_PER_CH), "start_ns": BLOCK_START,
                                "raw_readings": raw_count, "finite": np.isfinite(sampled).all(axis=1), **flags}))
    value_blocks.append(sampled)
    in_period = int(((t >= T_START) & (t < T_END)).sum())
    print(f"  {ch}: {in_period:>10,} raw readings in 2000-01 → 2002-10 | annotated blocks {flags['any_annotation'].sum():,}")

ESA = pd.concat(frames, ignore_index=True)
ESA_VALUES = np.concatenate(value_blocks)
ESA_EVENTS = (pd.concat(event_parts, ignore_index=True) if event_parts
              else pd.DataFrame(columns=["row", "event_id", "category"]))
ESA["start"] = pd.to_datetime(ESA["start_ns"])
ESA["split"] = np.select([ESA["start_ns"] < T_FIT_END, ESA["start_ns"] < T_CAL_END],
                         ["fit", "calibration"], "confirmation")
ESA["rare_only"] = ESA["primary"] & ~ESA["anomaly"]

# --- scaling: nominal fitting blocks only (all positive fitting blocks excluded) ---
fit_nominal = (ESA["split"].eq("fit") & ESA["finite"] & ~ESA["any_annotation"]).to_numpy()
ESA_SCALER = {}
for ch in ESA_CHANNELS:
    m = fit_nominal & ESA["channel"].eq(ch).to_numpy()
    vals = ESA_VALUES[m].ravel()
    std = float(vals.std(ddof=0))
    ESA_SCALER[ch] = {"mean": float(vals.mean()), "scale": std if std > 0 else 1.0,
                      "constant": bool(std == 0), "nominal_fit_blocks": int(m.sum())}
mean = ESA["channel"].map(lambda c: ESA_SCALER[c]["mean"]).to_numpy()
scale = ESA["channel"].map(lambda c: ESA_SCALER[c]["scale"]).to_numpy()
ESA_X = ((ESA_VALUES - mean[:, None]) / scale[:, None]).astype(np.float32)[..., None]

eligible = np.flatnonzero(fit_nominal)
FIT_ROWS = np.sort(np.random.default_rng(42).choice(eligible, size=min(N_FIT_ESA, len(eligible)), replace=False))
CAL_ROWS = np.flatnonzero((ESA["split"].eq("calibration") & ESA["finite"]).to_numpy())
CAL_NOMINAL_ROWS = CAL_ROWS[~ESA["primary"].to_numpy()[CAL_ROWS]]
CONF_ROWS = np.flatnonzero((ESA["split"].eq("confirmation") & ESA["finite"]).to_numpy())


def esa_representative_rows(rows):
    """Per-channel minimum and maximum windows, plus a seeded random fill to 967 (nominal fitting only)."""
    rng = np.random.default_rng(42)
    chosen, channel = set(), ESA["channel"].to_numpy()[rows]
    for ch in ESA_CHANNELS:
        r = rows[channel == ch]
        if len(r):
            x = ESA_X[r, :, 0]
            chosen.update([int(r[np.argmin(x.min(axis=1))]), int(r[np.argmax(x.max(axis=1))])])
    fixed = np.array(sorted(chosen))
    extra = rng.choice(np.setdiff1d(rows, fixed), size=N_REPRESENTATIVE - len(fixed), replace=False)
    return np.sort(np.concatenate([fixed, extra]))


conf = ESA.iloc[CONF_ROWS]
ev = ESA_EVENTS[ESA_EVENTS["row"].isin(CONF_ROWS)]
ev_rare = ev["category"].str.lower().str.contains("rare")
ev_anom = ev["category"].str.lower().eq("anomaly")
ESA_COHORT = {
    "Nominal fitting blocks available": int(len(eligible)),
    "Fitting windows used": int(len(FIT_ROWS)),
    "Calibration blocks": int(len(CAL_ROWS)),
    "Nominal calibration blocks": int(len(CAL_NOMINAL_ROWS)),
    "Confirmation channel-blocks": int(len(CONF_ROWS)),
    "Positive confirmation blocks": int(conf["primary"].sum()),
    "Rare-only positive blocks": int(conf["rare_only"].sum()),
    "Blocks with a true-anomaly annotation": int(conf["anomaly"].sum()),
    "Distinct rare-event IDs": int(ev.loc[ev_rare, "event_id"].nunique()),
    "Distinct true-anomaly IDs": int(ev.loc[ev_anom, "event_id"].nunique()),
    "Non-finite blocks excluded (all periods)": int((~ESA["finite"]).sum()),
}
PAPER_ESA = {"Fitting windows used": 29_242, "Calibration blocks": 18_216, "Nominal calibration blocks": 17_876,
             "Confirmation channel-blocks": 72_270, "Positive confirmation blocks": 3_275,
             "Rare-only positive blocks": 3_271, "Blocks with a true-anomaly annotation": 4,
             "Distinct rare-event IDs": 200, "Distinct true-anomaly IDs": 1}
print("\nESA cohort (✓ = same as the paper draft):")
for key, value in ESA_COHORT.items():
    if key in PAPER_ESA:
        check(key, value, PAPER_ESA[key])
    else:
        print(f"    {key}: {value:,}")
print("  True-anomaly IDs in confirmation:", sorted(ev.loc[ev_anom, "event_id"].unique()))

ESA.drop(columns=["start"]).to_csv(OUT / "data/esa_blocks.csv.gz", index=False)
ESA_EVENTS.to_csv(OUT / "data/esa_block_events.csv.gz", index=False)
np.savez_compressed(OUT / "data/esa_block_values.npz", values=ESA_VALUES)
(OUT / "data/esa_scaler.json").write_text(json.dumps(ESA_SCALER, indent=2))
(OUT / "data/esa_cohort.json").write_text(json.dumps(ESA_COHORT, indent=2))
