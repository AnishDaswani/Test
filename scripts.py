import os
import re
import sys
import glob
import shutil
import tokenize
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "staticfiles"}
EXTS = {".py", ".html", ".htm", ".js", ".css", ".md", ".txt", ".json", ".yml", ".yaml"}
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
JS_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
JS_LINE = re.compile(r"//.*?$", re.MULTILINE)


def cmd_check_model():
    try:
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
    except Exception as e:
        print("TensorFlow import failed:", e)
        return 1
    path = "earthsearch_preview_haze_model.keras"
    if os.path.exists(path):
        print("Found model at", path)
        try:
            m = tf.keras.models.load_model(path)
            m.summary()
        except Exception as e:
            print("Load failed:", e)
    else:
        print("Model not found at", path)
    return 0


def cmd_check_images():
    files = glob.glob("plots/**/*.png", recursive=True) + glob.glob("*.png")
    if not files:
        print("No PNG files found")
    else:
        for f in sorted(files):
            try:
                print(f, os.path.getsize(f))
            except Exception as e:
                print(f, "ERROR", e)
    return 0


def cmd_diagnose():
    sys.path.insert(0, str(ROOT))
    from ml_app import core
    import numpy as np
    model_file = "earthsearch_preview_haze_model.keras"
    collections = ["sentinel-2-l2a"]
    bbox = [-84.6, 33.7, -84.2, 34.1]
    date_range = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"
    page_limit = 3
    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    try:
        import tensorflow as tf
    except Exception:
        print("TensorFlow required for diagnose")
        return 1
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        print("Loaded", model_file)
    else:
        model = core.build_model(input_shape=(*core.IMG_SIZE, 3))
        print("Built fresh model")
    try:
        X = core.build_preview_dataset(collections, bbox, date_range, page_limit=page_limit)
        print("Fetched shape", X.shape)
    except Exception as e:
        print("Fetch failed:", e)
        X = np.random.randint(0, 255, (16, *core.IMG_SIZE, 3), dtype=np.uint8)
    y, _, _ = core.haze_proxy_labels(X)
    print("Proxy counts:", np.bincount(y))
    preds = model.predict(X.astype("float32") / 255.0, batch_size=8)
    probs = preds[:, 1]
    pred_class = np.argmax(preds, axis=1)
    print("Probs min/max/mean:", probs.min(), probs.max(), probs.mean())
    cm = np.zeros((2, 2), dtype=int)
    for gt, pr in zip(y, pred_class):
        if gt in (0, 1) and pr in (0, 1):
            cm[gt, pr] += 1
    print("Confusion:\n", cm)
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        yp = (probs > t).astype(int)
        print(f"  {t:.2f}  det={yp.mean():.3f}  acc={(yp == y).mean():.3f}")
    mis = [i for i, (gt, pr) in enumerate(zip(y, pred_class)) if gt != pr]
    gradcam_dir = out_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)
    try:
        from ml_app.core import make_gradcam_heatmap, overlay_gradcam_on_image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for i, idx in enumerate(mis[:6]):
            img = X[idx].astype("float32") / 255.0
            heat = make_gradcam_heatmap(img, model, core.IMG_SIZE)
            overlay = overlay_gradcam_on_image(X[idx].astype("uint8"), heat, alpha=0.5)
            out_path = gradcam_dir / f"mis_{i}_idx{idx}_gt{y[idx]}_pr{pred_class[idx]}.png"
            plt.imsave(str(out_path), overlay)
            print("Wrote", out_path)
    except Exception as e:
        print("Grad-CAM failed:", e)
    return 0


def cmd_eval():
    sys.path.insert(0, str(ROOT))
    from ml_app import core
    import numpy as np
    model_file = "earthsearch_preview_haze_model.keras"
    page_limit = 2
    bbox = [-84.6, 33.7, -84.2, 34.1]
    date_range = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"
    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow required:", e)
        return 1
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file)
        print("Loaded", model_file)
    else:
        model = core.build_model(input_shape=(*core.IMG_SIZE, 3))
        print("Built fresh model")
    try:
        X = core.build_preview_dataset(["sentinel-2-l2a"], bbox, date_range, page_limit=page_limit)
        print("Shape:", X.shape)
    except Exception as e:
        print("Fetch failed:", e)
        X = np.random.randint(0, 255, (8, *core.IMG_SIZE, 3), dtype=np.uint8)
    try:
        y, _, _ = core.haze_proxy_labels(X)
        print("Labels:", np.bincount(y))
    except Exception as e:
        print("Labels failed:", e)
        y = np.zeros(len(X), dtype=np.int64)
    preds = model.predict(X.astype("float32") / 255.0, batch_size=8)
    pred_class = np.argmax(preds, axis=1)
    acc = (pred_class == y).mean()
    print("Accuracy vs proxy:", acc)
    cm = np.zeros((2, 2), dtype=int)
    for gt, pr in zip(y, pred_class):
        if gt in (0, 1) and pr in (0, 1):
            cm[gt, pr] += 1
    print("Confusion:\n", cm)
    return 0


def cmd_fine_tune():
    sys.path.insert(0, str(ROOT))
    from ml_app import core
    import numpy as np
    model_in = "earthsearch_preview_haze_model.keras"
    model_out = "earthsearch_preview_haze_model_finetuned.keras"
    bbox = [-84.6, 33.7, -84.2, 34.1]
    date_range = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"
    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow required:", e)
        return 1
    if os.path.exists(model_in):
        model = tf.keras.models.load_model(model_in)
        print("Loaded", model_in)
    else:
        model = core.build_model(input_shape=(*core.IMG_SIZE, 3))
    try:
        X = core.build_preview_dataset(["sentinel-2-l2a"], bbox, date_range, page_limit=3)
        y, _, _ = core.haze_proxy_labels(X)
        print("Fetched", X.shape, "labels", np.bincount(y))
    except Exception as e:
        print("Fetch failed:", e)
        X = np.random.randint(0, 255, (80, *core.IMG_SIZE, 3), dtype=np.uint8)
        y = np.random.randint(0, 2, size=(80,))
    idxs = []
    for c in (0, 1):
        ci = np.where(y == c)[0]
        np.random.shuffle(ci)
        idxs.extend(ci[: min(len(ci), 40)].tolist())
    np.random.shuffle(idxs)
    X_sub = X[idxs]
    y_sub = y[idxs]
    if len(X_sub) < 8:
        print("Not enough samples")
        return 1
    n = len(X_sub)
    tr = int(0.8 * n)
    x_tr = X_sub[:tr]
    y_tr = y_sub[:tr]
    x_val = X_sub[tr:]
    y_val = y_sub[tr:]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_tr.astype("float32") / 255.0, y_tr, validation_data=(x_val.astype("float32") / 255.0, y_val), epochs=6, batch_size=8, verbose=2)
    preds = model.predict(X.astype("float32") / 255.0)
    acc = (np.argmax(preds, axis=1) == y).mean()
    print("Post-finetune accuracy:", acc)
    model.save(model_out)
    print("Saved", model_out)
    return 0


def cmd_cleanup():
    for d in [ROOT / "Test", ROOT / "__pycache__", ROOT / "staticfiles"]:
        if d.exists():
            print("Removing", d)
            shutil.rmtree(d, ignore_errors=True)
        else:
            print("Not found", d)
    return 0


def _strip_py_comments(src):
    out = io.StringIO()
    tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    prev_end = (1, 0)
    for toknum, tokval, start, end, _ in tokens:
        if toknum == tokenize.COMMENT:
            continue
        (srow, scol), (erow, ecol) = start, end
        if prev_end[0] < srow:
            out.write("\n" * (srow - prev_end[0]))
            prev_end = (srow, 0)
        if prev_end[1] < scol:
            out.write(" " * (scol - prev_end[1]))
        out.write(tokval)
        prev_end = (erow, ecol)
    return out.getvalue()


def cmd_remove_comments():
    modified = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            path = Path(dirpath) / fname
            if path.suffix.lower() not in EXTS:
                continue
            if any(p in SKIP_DIRS for p in path.parts):
                continue
            try:
                src = path.read_text(encoding="utf-8")
            except Exception:
                continue
            ext = path.suffix.lower()
            if ext == ".py":
                out = _strip_py_comments(src)
            elif ext in {".html", ".htm", ".md"}:
                out = HTML_COMMENT.sub("", src)
            elif ext in {".js", ".css"}:
                out = JS_LINE.sub("", JS_BLOCK.sub("", src))
            elif ext in {".json", ".yml", ".yaml", ".txt"}:
                out = "".join(line for line in src.splitlines(keepends=True) if not line.lstrip().startswith("#"))
            else:
                continue
            if out != src:
                path.write_text(out, encoding="utf-8")
                modified.append(str(path.relative_to(ROOT)))
    if modified:
        print("Modified:")
        for m in modified:
            print(" ", m)
    else:
        print("No files modified")
    return 0


def main():
    commands = {
        "diagnose": cmd_diagnose,
        "eval": cmd_eval,
        "check-model": cmd_check_model,
        "check-images": cmd_check_images,
        "fine-tune": cmd_fine_tune,
        "cleanup": cmd_cleanup,
        "remove-comments": cmd_remove_comments,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python scripts.py <command>")
        print("Commands:", ", ".join(commands))
        return 2
    return commands[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
