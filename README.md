## Pollution Detector (TensorFlow + Django)

This project is a small, production-style Django app plus a set of scripts for
training and serving a TensorFlow model that detects haze/pollution in
satellite imagery.

The model is trained on preview tiles from the Earth Search STAC API and served
through a simple web UI with:
- A home page showing model status and available plots
- A prediction page (single-image upload with optional Grad‑CAM)
- A training page with live progress
- A graphs page for viewing training curves and visualizations

Supported runtime: **Python 3.12**.

---

### 1. Quick start (Windows PowerShell)

From the repo root:

1. Activate the virtual environment (or create and use your own):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run migrations (first time only):

```powershell
python manage.py migrate
```

4. Run Django checks and start the server:

```powershell
python manage.py check
python manage.py runserver
```

Then open `http://127.0.0.1:8000/` in your browser.

---

### 2. Training and CLI tools

You can train and inspect the model from the command line:

- Train from STAC previews and write plots/model:

```powershell
python main.py
```

- Run helper commands:

```powershell
python scripts.py diagnose        # quick sanity check + Grad‑CAM on a small sample
python scripts.py eval            # small evaluation vs proxy labels
python scripts.py fine-tune       # short fine‑tune pass
python scripts.py check-model     # load and summarize the saved model
python scripts.py check-images    # list available PNG plots
python scripts.py cleanup         # remove transient build/output folders
```

The Django training page (`/train/`) uses the same core model and dataset
utilities as `main.py`, but runs the training loop in a background thread and
streams progress back as JSON.

---

### 3. Project layout

- `pollution_detector/` – Django settings, URLs, WSGI/ASGI entrypoints  
- `ml_app/core.py` – STAC data loading, image decoding, proxy labels, CNN, Grad‑CAM  
- `ml_app/views.py` – all HTTP views and JSON APIs  
- `ml_app/training.py` – background training logic and progress state  
- `ml_app/apps.py` – Django app config + optional `main.py` startup hook  
- `templates/ml_app/` – HTML templates for home, predict, train, graphs  
- `static/ml_app/` – CSS and JavaScript for the UI  
- `media/` – runtime media (prediction images, copied plots)  
- `plots/` – training plots and `skipped_assets.log`  
- `main.py` – standalone training script  
- `scripts.py` – CLI utilities

TensorFlow is imported lazily in the web code. The Django app can start
without TensorFlow installed, but **prediction, training, and Grad‑CAM**
require TensorFlow to be available in the environment.
