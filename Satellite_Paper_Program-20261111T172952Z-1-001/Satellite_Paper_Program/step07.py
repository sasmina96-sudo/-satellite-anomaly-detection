# STEP 7 — ESA: run the download (cached after the first run)
ESA_MANIFEST = fetch_esa_members()
(OUT / "data/esa_member_manifest.json").write_text(json.dumps(ESA_MANIFEST, indent=2))
print("Mode:", ESA_MANIFEST["mode"], "|", len(ESA_MANIFEST["members"]), "files verified")
