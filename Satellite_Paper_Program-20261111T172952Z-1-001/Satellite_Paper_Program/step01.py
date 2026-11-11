# STEP 1 — Setup: pinned packages, CPU-only deterministic TensorFlow, output folders
import os, sys, subprocess, importlib, importlib.metadata, json, platform, hashlib, time, re, zlib
import inspect, marshal, traceback, warnings, zipfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only, as in the paper
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


def installed_version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


need = []
if "2.20.0" not in (installed_version("tensorflow"), installed_version("tensorflow-cpu")):
    need.append("tensorflow==2.20.0")
if installed_version("ai-edge-litert") != "2.3.0":
    need.append("ai-edge-litert==2.3.0")
if installed_version("remotezip") is None:
    need.append("remotezip")
if need:
    print("Installing:", " ".join(need))
    installed_ok = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need]).returncode == 0
    heavy = any(p.split("==")[0] in ("tensorflow", "ai-edge-litert") for p in need)
    if not installed_ok:
        print("WARNING: install failed — continuing with the installed versions (they are recorded below).")
    elif heavy and any(m in sys.modules for m in ("numpy", "tensorflow", "keras")):
        print("\n>>> Packages changed. The runtime restarts now.")
        print(">>> When it reconnects: Runtime → Run all (again). <<<")
        time.sleep(2)
        os.kill(os.getpid(), 9)

import numpy as np
import pandas as pd
import tensorflow as tf
import absl.logging

tf.get_logger().setLevel("ERROR")
absl.logging.set_verbosity(absl.logging.ERROR)
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError as error:
    print("Note: thread settings were already fixed in this session:", error)
tf.config.experimental.enable_op_determinism()

try:
    from ai_edge_litert.interpreter import Interpreter as LiteInterpreter
    RUNTIME = f"LiteRT {installed_version('ai-edge-litert')}"
except Exception as error:
    LiteInterpreter = tf.lite.Interpreter
    RUNTIME = f"tf.lite.Interpreter (TensorFlow {tf.__version__}); LiteRT unavailable: {error}"

from pathlib import Path
from datetime import datetime, timezone

try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/Satellite_Full_Paper_Run")
except ImportError:
    BASE = Path.cwd() / "Satellite_Full_Paper_Run"
CACHE = BASE / "data_cache"                     # downloads are reused by later runs
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = BASE / "runs" / RUN_ID
for sub in ["data", "models/opssat", "models/esa", "results", "figures", "lock"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
for sub in ["opssat", "esa_mission2"]:
    (CACHE / sub).mkdir(parents=True, exist_ok=True)

# ---- Fixed protocol (same values as the paper) ----
SEEDS = [42, 43, 44]
ARCHS = ["lstm", "dense"]
NAME = {"lstm": "LSTM", "dense": "Dense"}
PARAMS = {"lstm": 3281, "dense": 428}
WINDOW, STRIDE = 8, 4
EPOCHS, BATCH, PATIENCE, LR, CLIPNORM = 30, 128, 5, 1e-3, 1.0
N_REPRESENTATIVE = 967
N_FIT_ESA = 29_242

cpu = platform.processor()
if Path("/proc/cpuinfo").exists():
    names = [l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
             if l.startswith("model name")]
    cpu = names[0] if names else cpu
ENV = {"run_id": RUN_ID, "python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
       **{k: installed_version(k) for k in ["numpy", "pandas", "scikit-learn", "tensorflow", "tensorflow-cpu",
                                            "keras", "ai-edge-litert", "remotezip", "matplotlib"]},
       "inference_runtime": RUNTIME, "intra_op_threads": 2, "inter_op_threads": 1,
       "op_determinism": True, "gpu": "disabled"}
(OUT / "results/environment.json").write_text(json.dumps(ENV, indent=2))
for key, value in ENV.items():
    if value:
        print(f"{key:>18}: {value}")
print("\nOutputs for this run:", OUT)
