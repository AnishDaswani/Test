#000000
# Pollution Detector using Tensorflow
### Requirements:
```
    Tensorflow
    Matplotlib
    Numpy
```
### Project Description:
```
    Python Tensorflow AI model to identify pollution through satellite images.
     Training conducted using earth search satellite images.
```
### Setup:

 **1. Run Venv**

 >. .\\.venv312\Scripts\Activate.ps1

 **2. Install Dependencies**

 >pip install tensorflow matplotlib numpy

 **3. Run**

````markdown
# Pollution Detector (TensorFlow + Django)

This repository contains a small Django web app and supporting scripts for
training/previewing a TensorFlow model that detects haze/pollution in
satellite imagery.

Supported runtime: Python 3.12 (the project was tested on 3.12.x).

Quick start (Windows PowerShell)

1. Activate the project virtualenv (created as `.venv` in the repo root):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run Django system checks:

```powershell
.\.venv\Scripts\python3.12 manage.py check
```

4. Start the development server:

```powershell
.\.venv\Scripts\python3.12 manage.py runserver
```

Notes

- The web app will copy `plots/*.png` into `media/plots/` when requested.
- Image decoding now performs a magic-bytes check before invoking TensorFlow
    decoders to avoid low-level "Unknown image file format" crashes.
- Unsupported or skipped preview assets are appended to `plots/skipped_assets.log`.
- TensorFlow is imported lazily in the web code; the app can start without
    TensorFlow installed, but prediction (model inference), Grad-CAM visualizations,
    and training require TensorFlow present in the environment.

If you want a quick smoke test, use the `.venv` python to run the system check
and start the server (steps 3–4 above). For predictable results run everything
inside the provided `.venv`.

````

