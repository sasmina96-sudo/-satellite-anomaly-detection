# STEP 14 — Write PAPER_NUMBERS.md and download one ZIP with all results
notes = [f"# Results for run {RUN_ID}", "",
         f"Environment: Python {ENV['python']}, TensorFlow {tf.__version__}, {RUNTIME}, CPU: {ENV['cpu']}", "",
         "All numbers below come from this run and replace the numbers in the current draft.", ""]
for title, table in TABLES.items():
    notes += [f"## {title}", "", md_table(table), ""]
notes += ["## Power — energy per inference (ESTIMATE, not measured)", "",
          md_table(POWER.drop(columns=["Graph ops"]).round(2)), "",
          "Arithmetic + weight-fetch estimate uses Horowitz (ISSCC 2014) 45 nm per-operation energies; it excludes "
          "activations, requantization, control, idle power and interpreter overhead. Latency-based column assumes "
          f"{P_ACTIVE_W:g} W active power on the Colab CPU.", "",
          "## Cost — downlink policies on the ESA confirmation year", "",
          md_table(DOWNLINK[show].round(3).reset_index()), "",
          md_table(COST.round(3)), "",
          f"Assumptions: {BITS_PER_READING}-bit readings; summary = {SUMMARY_BITS} bits/block; transmitter DC power "
          f"{TX_DC_POWER_W} W; link information rates {', '.join(INFO_RATES_BPS)}; headers, timestamps, compression and "
          "retransmissions not modelled; ground-station price "
          + (f"{GS_USD_PER_MINUTE} USD/min." if GS_USD_PER_MINUTE else "not applied (minutes reported)."), "",
          "## Lock", "", f"Lock file SHA-256: {sha256_file(LOCK_PATH)} ({len(LOCK['files'])} files hashed)."]
(OUT / "PAPER_NUMBERS.md").write_text("\n".join(notes), encoding="utf-8")
print("\n".join(notes[:6]))

bundle = Path("/content" if Path("/content").exists() else BASE) / f"satellite_paper_run_{RUN_ID}.zip"
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=str(path.relative_to(OUT)))
print(f"\nSaved {bundle} ({bundle.stat().st_size / 1e6:.1f} MB). Drive copy: {OUT}")
try:
    from google.colab import files
    files.download(str(bundle))
except ImportError:
    pass
