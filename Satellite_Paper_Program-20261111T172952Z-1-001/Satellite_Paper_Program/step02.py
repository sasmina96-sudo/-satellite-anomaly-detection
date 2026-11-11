# STEP 2 — Shared functions: models, conversion, inference, thresholds, metrics, baselines, timing
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

ROWS = []        # every metric row (cohort × target × model × seed × condition)
ARTIFACTS = []   # one row per training run: sizes, hashes, audits, timings, thresholds
CONDITION_NAMES = {"fp32_f1": "FP32, own max-F1 threshold", "int8_f1": "INT8, own max-F1 threshold",
                   "int8_transfer": "INT8, FP32 threshold transferred", "fp32_p99": "FP32, nominal P99",
                   "int8_p99": "INT8, nominal P99", "f1": "baseline, max-F1", "p99": "baseline, nominal P99"}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def check(name, actual, expected):
    same = actual == expected
    print(f"  {'✓' if same else '⚠'} {name}: {actual:,}" + ("" if same else f"   (paper draft: {expected:,})"))
    return same


def save_progress():
    pd.DataFrame(ROWS).to_csv(OUT / "results/all_metrics.csv", index=False)
    pd.DataFrame(ARTIFACTS).to_csv(OUT / "results/artifacts.csv", index=False)


# ---------- models ----------
def build_model(arch):
    L = tf.keras.layers
    inputs = tf.keras.Input(shape=(WINDOW, 1), name="telemetry")
    if arch == "lstm":
        h = L.LSTM(16, unroll=True)(inputs)                             # 16-unit unrolled encoder
        h = L.RepeatVector(WINDOW)(h)
        h = L.LSTM(16, return_sequences=True, unroll=True)(h)           # 16-unit unrolled decoder
        outputs = L.TimeDistributed(L.Dense(1))(h)                       # scalar linear output
    elif arch == "dense":
        h = L.Flatten(name="flatten")(inputs)
        h = L.Dense(16, activation="relu", name="encoder")(h)
        h = L.Dense(4, activation="linear", name="bottleneck")(h)
        h = L.Dense(16, activation="relu", name="decoder")(h)
        h = L.Dense(WINDOW, activation="linear", name="dense")(h)
        outputs = L.Reshape((WINDOW, 1), name="reconstruction")(h)
    else:
        raise ValueError(arch)
    # Names match the earlier notebooks: tensor names are stored inside .tflite files and change their size.
    model = tf.keras.Model(inputs, outputs, name=f"{arch}_ae")
    assert model.count_params() == PARAMS[arch], (arch, model.count_params())
    return model


def train_model(arch, seed, X_fit, X_val_nominal, run_dir):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(arch)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM), loss="mse")
    data = (tf.data.Dataset.from_tensor_slices((X_fit, X_fit))
            .shuffle(len(X_fit), seed=seed, reshuffle_each_iteration=True)   # explicit, seeded shuffling
            .batch(BATCH))
    best = run_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_loss", save_best_only=True, mode="min"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, mode="min"),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.time()
    history = model.fit(data, validation_data=(X_val_nominal, X_val_nominal), validation_batch_size=1024,
                        epochs=EPOCHS, callbacks=callbacks, verbose=0)
    loss, val = np.asarray(history.history["loss"]), np.asarray(history.history["val_loss"])
    if not (np.isfinite(loss).all() and np.isfinite(val).all()):
        raise RuntimeError("Non-finite training loss")
    model = tf.keras.models.load_model(best)
    return model, {"epochs_run": int(len(val)), "best_epoch": int(np.argmin(val)) + 1,
                   "best_val_mse": float(val.min()), "train_seconds": round(time.time() - start, 1)}


# ---------- conversion ----------
def freeze(model):
    @tf.function(input_signature=[tf.TensorSpec([1, WINDOW, 1], tf.float32, name="telemetry")])
    def serve(x):
        return model(x, training=False)
    frozen = convert_variables_to_constants_v2(serve.get_concrete_function())
    assert not [op.name for op in frozen.graph.get_operations() if op.type == "ReadVariableOp"]
    return frozen


def convert_fp32(frozen):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def convert_int8(frozen, representative, policy):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([s[None, ...].astype(np.float32)] for s in representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    if policy == "legacy_per_channel":          # accepted policy (paper amendment)
        converter.experimental_new_quantizer = False
        converter._experimental_disable_per_channel = False
    elif policy != "default":                   # "default" = diagnostic only, never scored
        raise ValueError(policy)
    return converter.convert()


def audit(model_bytes):
    it = LiteInterpreter(model_content=model_bytes)
    it.allocate_tensors()
    details = it.get_tensor_details()
    floats = [t["name"] for t in details if np.issubdtype(t["dtype"], np.floating)]
    return {"n_tensors": len(details), "float_tensors": len(floats), "float_tensor_names": floats[:20],
            "dtype_counts": dict(Counter(np.dtype(t["dtype"]).name for t in details)),
            "input_dtype": np.dtype(it.get_input_details()[0]["dtype"]).name,
            "output_dtype": np.dtype(it.get_output_details()[0]["dtype"]).name}


# ---------- inference ----------
class Runner:
    """Batch-1 LiteRT inference. INT8: round + clip inputs, dequantize outputs."""

    def __init__(self, model_bytes):
        self.it = LiteInterpreter(model_content=model_bytes, num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.is_int8 = self.inp["dtype"] == np.int8

    def prepare(self, X):
        X = np.asarray(X, dtype=np.float32)
        if not self.is_int8:
            return X, np.zeros(len(X), dtype=np.int64)
        scale, zero = self.inp["quantization"]
        q = np.rint(X.astype(np.float64) / scale + zero)
        clipped = ((q < -128) | (q > 127)).reshape(len(X), -1).sum(axis=1)
        return np.clip(q, -128, 127).astype(np.int8), clipped

    def run(self, X):
        data, clipped = self.prepare(X)
        out = np.empty((len(data), *[int(s) for s in self.out["shape"][1:]]), dtype=np.float64)
        set_tensor, invoke, get_tensor = self.it.set_tensor, self.it.invoke, self.it.get_tensor
        i_in, i_out = self.inp["index"], self.out["index"]
        for i in range(len(data)):
            set_tensor(i_in, data[i:i + 1])
            invoke()
            out[i] = get_tensor(i_out)[0]
        if self.is_int8:
            scale, zero = self.out["quantization"]
            out = (out - zero) * scale
        assert np.isfinite(out).all(), "Non-finite model output"
        return out, clipped


def window_mse(X, recon):
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    return ((X - recon.reshape(len(X), -1)) ** 2).mean(axis=1)


def check_agreement(model, frozen, fp32_bytes, X_check):
    keras_out = model.predict(X_check, batch_size=128, verbose=0).astype(np.float64)
    frozen_out = np.concatenate([frozen(tf.convert_to_tensor(x[None, ...]))[0].numpy() for x in X_check], axis=0)
    lite_out, _ = Runner(fp32_bytes).run(X_check)
    np.testing.assert_allclose(frozen_out, keras_out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(lite_out, keras_out, rtol=1e-4, atol=1e-5)
    return float(np.abs(lite_out - keras_out).max())


# ---------- thresholds & metrics ----------
def max_f1_threshold(labels, scores):
    p, r, thr = precision_recall_curve(labels, scores)
    f = np.divide(2 * p[:-1] * r[:-1], p[:-1] + r[:-1], out=np.zeros(len(thr)), where=(p[:-1] + r[:-1]) > 0)
    return float(thr[np.flatnonzero(f == f.max())[-1]])        # exact ties → highest threshold


def p99_threshold(nominal_scores):
    return float(np.percentile(nominal_scores, 99, method="higher"))


def evaluate(labels, scores, threshold):
    labels, scores = np.asarray(labels).astype(int), np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)                 # score ≥ threshold → alarm
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    both = len(np.unique(labels)) == 2
    return {"threshold": float(threshold),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)) if both else np.nan,
            "average_precision": float(average_precision_score(labels, scores)) if both else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(labels))}


def threshold_set(labels, scores_by_format):
    nominal = np.asarray(labels) == 0
    return {"fp32_f1": max_f1_threshold(labels, scores_by_format["fp32"]),
            "int8_f1": max_f1_threshold(labels, scores_by_format["int8"]),
            "fp32_p99": p99_threshold(scores_by_format["fp32"][nominal]),
            "int8_p99": p99_threshold(scores_by_format["int8"][nominal])}


def conditions(thr):
    return {"fp32_f1": ("fp32", thr["fp32_f1"]), "int8_f1": ("int8", thr["int8_f1"]),
            "int8_transfer": ("int8", thr["fp32_f1"]),
            "fp32_p99": ("fp32", thr["fp32_p99"]), "int8_p99": ("int8", thr["int8_p99"])}


# ---------- fixed statistical controls ----------
class Baselines:
    NAMES = ("PCA4", "ZScore", "Persistence")

    def __init__(self, mean, components):
        self.mean, self.components = np.asarray(mean), np.asarray(components)

    @classmethod
    def fit(cls, X_nominal):
        flat = np.asarray(X_nominal, dtype=np.float64).reshape(len(X_nominal), -1)
        pca = PCA(n_components=4, svd_solver="full").fit(flat)
        return cls(pca.mean_, pca.components_)

    @classmethod
    def load(cls, path):
        saved = np.load(path)
        return cls(saved["mean"], saved["components"])

    def save(self, path):
        np.savez(path, mean=self.mean, components=self.components)

    def scores(self, X):
        flat = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        recon = (flat - self.mean) @ self.components.T @ self.components + self.mean
        previous = np.concatenate([flat[:, :1], flat[:, :-1]], axis=1)   # persistence: copy predecessor
        return {"PCA4": ((flat - recon) ** 2).mean(axis=1),
                "ZScore": np.abs(flat).max(axis=1),
                "Persistence": ((flat - previous) ** 2).mean(axis=1)}


# ---------- timing & graph accounting ----------
def latency_benchmark(model_files, X_bench, repeats=6, warmup=100):
    runners = {}
    for fmt, model_bytes in model_files.items():
        r = Runner(model_bytes)
        data, _ = r.prepare(X_bench)
        for i in range(warmup):
            r.it.set_tensor(r.inp["index"], data[i % len(data):i % len(data) + 1]); r.it.invoke()
        runners[fmt] = (r, data)
    records, names = [], list(runners)
    for rep in range(repeats):                                   # alternate order to reduce bias
        for fmt in (names if rep % 2 == 0 else names[::-1]):
            r, data = runners[fmt]
            it, i_in, i_out = r.it, r.inp["index"], r.out["index"]
            for i in range(len(data)):
                t0 = time.perf_counter_ns()
                it.set_tensor(i_in, data[i:i + 1]); it.invoke(); it.get_tensor(i_out)
                records.append((fmt, (time.perf_counter_ns() - t0) / 1000))
    df = pd.DataFrame(records, columns=["format", "latency_us"])
    summary = df.groupby("format")["latency_us"].agg(median_us="median", mean_us="mean",
                                                     p95_us=lambda v: v.quantile(0.95))
    return summary.to_dict("index")


def count_macs(model_bytes):
    """Multiply-accumulates per inference, counted from the exported LiteRT graph."""
    it = LiteInterpreter(model_content=model_bytes)
    it.allocate_tensors()
    shapes = {d["index"]: tuple(int(s) for s in d["shape"]) for d in it.get_tensor_details()}
    total, ops = 0, Counter()
    for op in it._get_ops_details():
        name = op["op_name"]
        ops[name] += 1
        inputs = [int(i) for i in op["inputs"] if int(i) >= 0]
        if name == "FULLY_CONNECTED":
            n_out, n_in = shapes[inputs[1]][:2]
            total += (int(np.prod(shapes[inputs[0]])) // n_in) * n_out * n_in
        elif name == "BATCH_MATMUL":
            total += int(np.prod(shapes[int(op["outputs"][0])])) * shapes[inputs[0]][-1]
    return int(total), dict(ops)


def code_fingerprint(objects):
    digest = hashlib.sha256()
    for obj in objects:
        try:
            text = inspect.getsource(obj)
        except (OSError, TypeError):
            funcs = [f for f in vars(obj).values() if inspect.isfunction(f)] if inspect.isclass(obj) else [obj]
            text = "".join(marshal.dumps(f.__code__).hex() for f in funcs)
        digest.update(text.encode())
    return digest.hexdigest()


# ---------- one complete training run (shared by both cohorts) ----------
def train_and_export(arch, seed, cohort, X_fit, X_val_nominal, X_representative, X_check, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    model, info = train_model(arch, seed, X_fit, X_val_nominal, run_dir)
    frozen = freeze(model)
    fp32 = convert_fp32(frozen)
    info["fp32_max_abs_diff_vs_keras"] = check_agreement(model, frozen, fp32, X_check)
    try:                                       # diagnostic only — retained, never scored
        diag = audit(convert_int8(frozen, X_representative, "default"))
        info["default_quantizer_float_tensors"] = diag["float_tensors"]
    except Exception as error:
        info["default_quantizer_float_tensors"] = f"failed: {type(error).__name__}"
    int8 = convert_int8(frozen, X_representative, "legacy_per_channel")
    report = audit(int8)
    (run_dir / "int8_tensor_audit.json").write_text(json.dumps(report, indent=2))
    if report["float_tensors"] or report["input_dtype"] != "int8" or report["output_dtype"] != "int8":
        (run_dir / "REJECTED_model_int8.tflite").write_bytes(int8)
        raise RuntimeError(f"INT8 export rejected: {report['float_tensors']} floating tensors, "
                           f"io {report['input_dtype']}/{report['output_dtype']}")
    (run_dir / "model_fp32.tflite").write_bytes(fp32)
    (run_dir / "model_int8.tflite").write_bytes(int8)
    info.update(cohort=cohort, arch=arch, seed=seed, params=PARAMS[arch],
                fp32_bytes=len(fp32), int8_bytes=len(int8),
                fp32_sha256=sha256_bytes(fp32), int8_sha256=sha256_bytes(int8),
                int8_float_tensors=report["float_tensors"], int8_tensors=report["n_tensors"])
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return {"fp32": fp32, "int8": int8}, info


EVALUATOR_OBJS = [Runner, window_mse, max_f1_threshold, p99_threshold, evaluate, conditions, Baselines]
print("Functions ready.")
