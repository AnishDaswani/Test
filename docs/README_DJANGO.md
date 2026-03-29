# Django Web App - Pollution Detector

A Django-based web application for viewing ML model results and making predictions
on satellite imagery.

Supported runtime: Python 3.12 (use the project `.venv`).

## Quick Setup (Windows PowerShell)

1. Activate the project virtualenv:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run Django system checks and migrations (first run):

```powershell
.\.venv\Scripts\python3.12.exe manage.py check
.\.venv\Scripts\python3.12.exe manage.py migrate
```

4. Start the development server:

```powershell
.\.venv\Scripts\python3.12.exe manage.py runserver
```

Access the app at `http://127.0.0.1:8000/`.

## Important Notes & Behavior

- The app performs a magic-bytes check on uploaded / remote preview images before
   calling TensorFlow decoders. This prevents low-level TF errors for unknown
   image bytes (e.g. some JP2/JPEG2000 variants).
- Unsupported or skipped preview assets are logged to `plots/skipped_assets.log`.
- TensorFlow is imported lazily in the Django app so the server can start even
   if TensorFlow isn't installed. However, model inference, Grad-CAM visualizations,
   and training require TensorFlow present in the environment.
- For Grad-CAM visualizations the app saves generated images into `media/predictions/`.

## Endpoints

- Home: `/` — index and model status
- Graphs: `/graphs/` — plots and visualizations
- Predict: `/predict/` — upload single image and get predictions with optional Grad-CAM

## Project Layout

```
.
├── pollution_detector/     # Django project settings
├── ml_app/                 # Django app: views, training utilities, gradcam
├── templates/ml_app/       # HTML templates
├── static/                 # CSS and JS
├── media/                  # Uploaded/generated media (predictions, plots)
├── plots/                  # Source plot files (skipped_assets.log is written here)
├── manage.py
└── requirements.txt
```

If you want, I can add a small admin page to inspect `plots/skipped_assets.log`.
3. Plots are organized by category (Training, Confusion Matrix, Visualizations)

