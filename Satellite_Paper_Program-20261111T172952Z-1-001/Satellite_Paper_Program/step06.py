# STEP 6 — ESA download functions: fetch only the needed members of the 4.1 GB archive
import requests
import struct

ESA_URL = "https://zenodo.org/records/15237121/files/ESA-Mission2.zip?download=1"
ESA_ARCHIVE_MD5 = "bfc72012691427d9327eb41f726ce45e"    # complete-archive MD5 (checked only on full download)
ESA_DIR = CACHE / "esa_mission2"
ESA_CHANNELS = [f"channel_{i}" for i in range(18, 29)]   # published lightweight subset, Mission 2
WANTED = {"labels.csv": r"(^|/)labels\.csv$",
          "anomaly_types.csv": r"(^|/)anomaly_types\.csv$",
          "channels.csv": r"(^|/)channels\.csv$"}
WANTED.update({f"{ch}.zip": rf"(^|/)channels/{ch}(\.zip|\.pkl|\.pickle)?$" for ch in ESA_CHANNELS})
ALLOW_FULL_DOWNLOAD = True     # fallback if the server refuses range requests (downloads 4.1 GB once)
SUPPORTED_METHODS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}


def read_member(archive, info):
    """Read one archive member. Deflate64 (method 9, written by Windows) is inflated with inflate64."""
    if info.compress_type in SUPPORTED_METHODS:
        return archive.read(info)
    if info.compress_type != 9:
        raise NotImplementedError(f"{info.filename}: ZIP compression method {info.compress_type} is not supported")
    try:
        import inflate64
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "inflate64"], check=True)
        import inflate64
    archive.fp.seek(info.header_offset)
    header = archive.fp.read(30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    archive.fp.seek(info.header_offset + 30 + name_len + extra_len)
    return inflate64.Inflater().inflate(archive.fp.read(info.compress_size))


def open_remote_zip(url):
    """Open the remote ZIP with HTTP range requests (only the directory + chosen members are downloaded)."""
    from remotezip import RemoteZip
    last_error = None
    for suffix_range in (True, False):
        try:
            return RemoteZip(url, support_suffix_range=suffix_range, timeout=120,
                             headers={"User-Agent": "SatelliteResearch/1.0"})
        except Exception as error:
            last_error = error
    raise last_error


def open_full_download(url):
    target = (Path("/content") if Path("/content").exists() else ESA_DIR) / "ESA-Mission2.zip"
    if not target.exists():
        part, md5 = target.with_suffix(".part"), hashlib.md5()
        with requests.get(url, stream=True, timeout=300, headers={"User-Agent": "SatelliteResearch/1.0"}) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk)
                    md5.update(chunk)
        if md5.hexdigest() != ESA_ARCHIVE_MD5:
            raise ValueError("Full-archive MD5 mismatch")
        part.rename(target)
    return zipfile.ZipFile(target)


def fetch_esa_members():
    manifest_path = ESA_DIR / "member_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all((ESA_DIR / m["saved_as"]).exists() and sha256_file(ESA_DIR / m["saved_as"]) == m["sha256"]
               for m in manifest["members"]):
            print("Using cached ESA files (SHA-256 re-verified).")
            return manifest
    try:
        archive, mode = open_remote_zip(ESA_URL), "HTTP range requests (selected members only)"
    except Exception as error:
        if not ALLOW_FULL_DOWNLOAD:
            raise
        print("Range requests failed:", error, "→ downloading the full archive instead")
        archive, mode = open_full_download(ESA_URL), "full archive download (MD5 verified)"
    members = []
    with archive:
        names = archive.namelist()
        print(f"Archive lists {len(names):,} members, e.g. {names[:3]}")
        for saved_as, pattern in WANTED.items():
            hits = [n for n in names if re.search(pattern, n) and not n.endswith("/")]
            if not hits:
                if saved_as == "channels.csv":
                    continue                                   # optional metadata
                raise FileNotFoundError(f"No archive member matches {pattern}")
            if len(hits) > 1:
                raise RuntimeError(f"Ambiguous members for {saved_as}: {hits}")
            info = archive.getinfo(hits[0])
            data = read_member(archive, info)                 # CRC and size re-checked just below
            crc = zlib.crc32(data) & 0xFFFFFFFF
            if crc != info.CRC or len(data) != info.file_size:
                raise ValueError(f"CRC/size mismatch: {hits[0]}")
            (ESA_DIR / saved_as).write_bytes(data)
            members.append({"member": hits[0], "saved_as": saved_as, "bytes": len(data),
                            "crc32": f"{crc:08x}", "sha256": sha256_bytes(data)})
            print(f"  ✓ {hits[0]}  ({len(data) / 1e6:.1f} MB, CRC matches archive)")
    manifest = {"url": ESA_URL, "mode": mode, "complete_archive_md5_verified": mode.startswith("full"),
                "members": members, "fetched_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


print("ESA download functions ready.")
