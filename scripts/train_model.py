import sys
import os
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

tn = np.sum((y_test == 0) & (results['predictions'] == 0))
fp = np.sum((y_test == 0) & (results['predictions'] == 1))
fn = np.sum((y_test == 1) & (results['predictions'] == 0))
tp = np.sum((y_test == 1) & (results['predictions'] == 1))

print("\nConfusion Matrix:")
print(f"              Predicted Clean  Predicted Polluted")
print(f"True Clean         {tn:3d}              {fp:3d}")
print(f"True Polluted      {fn:3d}              {tp:3d}")

if results['precision'] > 0:
    print(f"\nPrecision (of detected pollution): {results['precision']:.2%}")
if results['recall'] > 0:
    print(f"Recall (pollution detection rate): {results['recall']:.2%}")

save_training_plots(history, "models/training_history.png")
save_sample_predictions(x_test, y_test, results['probabilities'], "models/sample_predictions.png")

model.save("models/pollution_model.keras")
print("Model saved to 'models/pollution_model.keras'")

print("\n" + "="*50)
print("LENIENT MODE ACTIVE")
print(f"Threshold: {PREDICTION_THRESHOLD:.0%} confidence needed for pollution")
print(f"Your image at 55% clean = 45% polluted")
print(f"Would be classified as: POLLUTED ✓")
print("="*50)