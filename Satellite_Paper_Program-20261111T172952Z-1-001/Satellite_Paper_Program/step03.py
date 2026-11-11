# STEP 3 — OPS-SAT: download, verify, split, scale and window (same pipeline as the paper)
import urllib.request

OPS_DIR = CACHE / "opssat"
OPS_FILES = {"segments.csv": "72f109630abb933a386106897a631188",
             "dataset.csv": "5246fdc5e4630a4cecbf7fb6bc8b795e"}
for name, md5 in OPS_FILES.items():
    path = OPS_DIR / name
    if not path.exists():
        print("Downloading", name)
        request = urllib.request.Request(f"https://zenodo.org/records/15108715/files/{name}?download=1",
                                         headers={"User-Agent": "SatelliteResearch/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != md5:
            raise ValueError(f"Download checksum failed: {name}")
        path.write_bytes(content)
    if hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f"Checksum mismatch: {name}")
    print(f"✓ {name} verified (MD5 {md5})")

segments = pd.read_csv(OPS_DIR / "segments.csv")
metadata = pd.read_csv(OPS_DIR / "dataset.csv")

# --- integrity checks ---
keys = ["channel", "segment"]
assert not metadata.duplicated(keys).any(), "duplicate segment records"
grouped = segments.groupby(keys)
assert grouped[["anomaly", "train"]].nunique().eq(1).all().all(), "conflicting labels inside a segment"
observed = grouped.agg(n=("value", "size"), a=("anomaly", "first"), t=("train", "first")).reset_index()
merged = metadata.merge(observed, on=keys, how="outer", validate="one_to_one", indicator=True)
assert merged["_merge"].eq("both").all()
assert merged["anomaly"].eq(merged["a"]).all() and merged["train"].eq(merged["t"]).all()
assert merged["len"].eq(merged["n"]).all() and np.isfinite(segments["value"].to_numpy(float)).all()
print("✓ Segment integrity checks passed")

# --- development split: ~20% of each channel/class of the original training partition (whole segments) ---
rng = np.random.default_rng(42)
split = metadata[["channel", "segment", "anomaly", "train"]].copy()
split["subset"] = np.where(split["train"].eq(1), "train", "test")
for _, group in split[split["train"].eq(1)].groupby(["channel", "anomaly"], sort=True):
    idx = group.sort_values("segment").index.to_numpy()
    if len(idx) < 2:
        continue
    n_val = min(len(idx) - 1, max(1, int(np.ceil(0.20 * len(idx)))))
    split.loc[rng.permutation(idx)[:n_val], "subset"] = "validation"
split["use_for_ae_fit"] = split["subset"].eq("train") & split["anomaly"].eq(0)
split.to_csv(OUT / "data/opssat_segment_split.csv", index=False)

# --- scaling: nominal fitting readings only ---
prepared = segments.merge(split[["channel", "segment", "subset", "use_for_ae_fit"]],
                          on=keys, how="left", validate="many_to_one")
OPS_SCALER = {}
for ch, g in prepared.groupby("channel"):
    v = g.loc[g["use_for_ae_fit"], "value"].to_numpy(dtype=np.float64)
    std = float(v.std(ddof=0))
    OPS_SCALER[ch] = {"mean": float(v.mean()), "scale": std if std > 0 else 1.0, "training_readings": int(len(v))}
prepared["scaled_value"] = ((prepared["value"] - prepared["channel"].map(lambda c: OPS_SCALER[c]["mean"]))
                            / prepared["channel"].map(lambda c: OPS_SCALER[c]["scale"]))
prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="raise")
prepared = prepared.sort_values(["channel", "segment", "timestamp"]).reset_index(drop=True)
(OUT / "data/opssat_scaler.json").write_text(json.dumps(OPS_SCALER, indent=2))

# --- windows: 8 readings, stride 4, final complete window included, no padding ---
windows, records = [], []
for (ch, seg), g in prepared.groupby(["channel", "segment"], sort=True):
    g = g.sort_values("timestamp")
    values = g["scaled_value"].to_numpy(dtype=np.float32)
    starts = list(range(0, len(values) - WINDOW + 1, STRIDE))
    if starts[-1] != len(values) - WINDOW:
        starts.append(len(values) - WINDOW)
    first = g.iloc[0]
    for s in starts:
        windows.append(values[s:s + WINDOW, None])
        records.append({"channel": ch, "segment": int(seg), "start_index": s, "subset": first["subset"],
                        "segment_anomaly": int(first["anomaly"]), "use_for_ae_fit": bool(first["use_for_ae_fit"])})
X_all, info_all = np.stack(windows), pd.DataFrame(records)

print("\nCounts (✓ = same as the paper draft):")
check("Readings", len(segments), 303_493)
check("Segments", len(metadata), 2_123)
check("Development fitting segments", int(split["subset"].eq("train").sum()), 1_268)
check("Nominal fitting segments", int(split["use_for_ae_fit"].sum()), 1_015)
check("Validation segments", int(split["subset"].eq("validation").sum()), 326)
check("Anomalous validation segments", int((split["subset"].eq("validation") & split["anomaly"].eq(1)).sum()), 68)
check("Test segments (never scored here)", int(split["subset"].eq("test").sum()), 529)
check("Nominal fitting windows", int(info_all["use_for_ae_fit"].sum()), 29_242)
check("Nominal validation windows", int((info_all["subset"].eq("validation") & info_all["segment_anomaly"].eq(0)).sum()), 8_385)
check("Validation windows", int(info_all["subset"].eq("validation").sum()), 12_135)

# Test windows are dropped here: fresh runs never see the archived test set.
keep = info_all["subset"].ne("test").to_numpy()
OPS_X, OPS_INFO = X_all[keep], info_all.loc[keep].reset_index(drop=True)
del X_all, info_all, windows, records
OPS_X_FIT = OPS_X[OPS_INFO["use_for_ae_fit"].to_numpy()]
val_mask = OPS_INFO["subset"].eq("validation").to_numpy()
OPS_X_VAL = OPS_X[val_mask]
OPS_VAL_INFO = OPS_INFO.loc[val_mask, ["channel", "segment", "segment_anomaly"]].reset_index(drop=True)
OPS_X_VAL_NOM = OPS_X[val_mask & OPS_INFO["segment_anomaly"].eq(0).to_numpy()]
print("\nDevelopment windows kept:", OPS_X.shape, "| test windows discarded")
