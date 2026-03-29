import os
import sys
import subprocess
import django
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.pollution_detector import *

BBOX = [-84.6, 33.7, -84.2, 34.1]
DATE_RANGE = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"

try:
    X_all = load_previews(PRIMARY_COLLECTIONS, BBOX, DATE_RANGE)
except Exception as e:
    print(f"Primary collections failed: {e}")
    X_all = load_previews(FALLBACK_COLLECTIONS, BBOX, DATE_RANGE)

y_all, scores = pollution_proxy_labels(X_all, POLLUTION_LABEL_THRESHOLD)

print(f"Loaded {len(X_all)} images")
print(f"Pollution labels: {np.sum(y_all)} polluted ({100*np.sum(y_all)/len(y_all):.1f}%), {len(y_all)-np.sum(y_all)} clean")

idx = np.random.permutation(len(X_all))
n = len(idx)
tr, va = int(0.7*n), int(0.85*n)

x_train, y_train = X_all[idx[:tr]], y_all[idx[:tr]]
x_val, y_val = X_all[idx[tr:va]], y_all[idx[tr:va]]
x_test, y_test = X_all[idx[va:]], y_all[idx[va:]]

model = build_model()
print(model.summary())

history = train_model(model, x_train, y_train, x_val, y_val, EPOCHS)

model.save('models/earthsearch_preview_haze_model.keras')

results = evaluate_model(model, x_test, y_test)

print(f"\n{'='*50}")
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Pollution Detection Rate: {results['pollution_rate']:.2%}")
print(f"True Pollution Rate: {results['true_pollution_rate']:.2%}")
print(f"Threshold used: {PREDICTION_THRESHOLD:.2%}")
print(f"{'='*50}")

print(f"\nConfidence Distribution:")
print(f"  Min pollution confidence: {results['probabilities'][:, 1].min():.3f}")
print(f"  Max pollution confidence: {results['probabilities'][:, 1].max():.3f}")
print(f"  Mean pollution confidence: {results['probabilities'][:, 1].mean():.3f}")
print(f"  Images with >35% pollution confidence: {np.sum(results['probabilities'][:, 1] > 0.35)}/{len(results['probabilities'])}")
print(f"  Images with >25% pollution confidence: {np.sum(results['probabilities'][:, 1] > 0.25)}/{len(results['probabilities'])}")

model_path = 'models/earthsearch_preview_haze_model.keras'
output_dir = 'web/static/models'

os.makedirs(output_dir, exist_ok=True)

if os.path.exists(model_path):
    try:
        subprocess.run([
            sys.executable, '-m', 'tensorflowjs_converter',
            '--input_format=keras',
            model_path,
            output_dir
        ], check=True)
        print(f"Model converted and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        print("Make sure tensorflowjs is installed: pip install tensorflowjs")
else:
    print(f"Model not found at {model_path}")

BASE_DIR = Path(__file__).resolve().parent.parent / 'web'
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_detector.settings')
django.setup()

from django.core.management import execute_from_command_line

execute_from_command_line(['manage.py', 'runserver', '8000'])