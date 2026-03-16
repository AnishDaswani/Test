import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_app.core import build_preview_dataset, build_model, make_tf_ds

PRIMARY_COLLECTIONS = ["sentinel-2-l2a"]
FALLBACK_COLLECTIONS = ["naip"]
BBOX = [-84.6, 33.7, -84.2, 34.1]
DATE_RANGE = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100
POLLUTION_LABEL_THRESHOLD = 40
PREDICTION_THRESHOLD = 0.35


def pollution_proxy_labels(X, percentile_thresh=40):
    x = tf.cast(X, tf.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    brown_gray = tf.reduce_mean((r + g) / (b + 0.05), axis=[1, 2])
    grayness = 1.0 - tf.math.reduce_std(x, axis=[1, 2, 3])
    color_score = brown_gray + grayness * 2
    mx, mn = tf.reduce_max(x, axis=-1), tf.reduce_min(x, axis=-1)
    sat = (mx - mn) / (mx + 1e-6)
    low_sat = (1.0 - tf.reduce_mean(sat, axis=[1, 2])) * 3
    gray = tf.image.rgb_to_grayscale(x)
    std = tf.math.reduce_std(gray, axis=[1, 2, 3])
    low_contrast = 2.0 / (std + 0.05)
    haze = tf.reduce_mean(gray, axis=[1, 2, 3]) * low_contrast
    color_std = tf.math.reduce_std(x, axis=[1, 2])
    uniformity = 1.0 / (tf.reduce_mean(color_std, axis=-1) + 0.05)
    score = (0.25 * color_score + 0.25 * low_sat + 0.20 * low_contrast + 0.15 * haze + 0.15 * uniformity).numpy()
    thresh = np.percentile(score, percentile_thresh)
    return (score > thresh).astype(np.int64), score


try:
    X_all = build_preview_dataset(PRIMARY_COLLECTIONS, BBOX, DATE_RANGE, page_limit=100, img_size=IMG_SIZE)
except Exception:
    X_all = build_preview_dataset(FALLBACK_COLLECTIONS, BBOX, DATE_RANGE, page_limit=100, img_size=IMG_SIZE)

y_all, _ = pollution_proxy_labels(X_all, POLLUTION_LABEL_THRESHOLD)
print(f"Loaded {len(X_all)} images; polluted: {np.sum(y_all)} ({100 * np.sum(y_all) / len(y_all):.1f}%)")

idx = np.random.permutation(len(X_all))
n = len(idx)
tr = int(0.7 * n)
va = int(0.85 * n)
x_train = X_all[idx[:tr]]
y_train = y_all[idx[:tr]]
x_val = X_all[idx[tr:va]]
y_val = y_all[idx[tr:va]]
x_test = X_all[idx[va:]]
y_test = y_all[idx[va:]]

model = build_model(input_shape=(*IMG_SIZE, 3), learning_rate=5e-4, dropout_rate=0.25, use_focal_loss=True)
train_ds = make_tf_ds(x_train, y_train, batch=BATCH_SIZE, shuffle=True)
val_ds = make_tf_ds(x_val, y_val, batch=BATCH_SIZE, shuffle=False)
early = tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss")
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early, reduce_lr],
    class_weight={0: 0.7, 1: 1.3},
    verbose=2,
)

probs = model.predict(x_test, batch_size=32)
y_pred = (probs[:, 1] > PREDICTION_THRESHOLD).astype(int)
acc = np.mean(y_pred == y_test)
print(f"\n{'=' * 50}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Pollution Detection: {np.sum(y_pred) / len(y_pred):.2%}")
print(f"True: {np.sum(y_test) / len(y_test):.2%}")
print(f"Threshold: {PREDICTION_THRESHOLD:.0%}")
print(f"{'=' * 50}")
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))
tp = np.sum((y_test == 1) & (y_pred == 1))
print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
if tp + fp > 0:
    print(f"Precision: {tp / (tp + fp):.2%}")
if tp + fn > 0:
    print(f"Recall: {tp / (tp + fn):.2%}")

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle("Sample Predictions with Confidence", fontsize=14)
clean_idx = np.where(probs[:, 1] < 0.25)[0]
border_idx = np.where((probs[:, 1] >= 0.25) & (probs[:, 1] < 0.65))[0]
poll_idx = np.where(probs[:, 1] >= 0.65)[0]
for i in range(5):
    if i < len(clean_idx):
        j = clean_idx[i]
        axes[0, i].imshow(x_test[j])
        axes[0, i].set_title(f"Clean: {probs[j][1]:.1%}", fontsize=10)
    axes[0, i].axis('off')
    if i < len(border_idx):
        j = border_idx[i]
        axes[1, i].imshow(x_test[j])
        axes[1, i].set_title(f"Border: {probs[j][1]:.1%}", fontsize=10, color='orange')
    axes[1, i].axis('off')
    if i < len(poll_idx):
        j = poll_idx[i]
        axes[2, i].imshow(x_test[j])
        axes[2, i].set_title(f"Polluted: {probs[j][1]:.1%}", fontsize=10, color='red')
    axes[2, i].axis('off')
plt.tight_layout()
plt.savefig("lenient_pollution_detection.png", dpi=150, bbox_inches='tight')
print("Saved lenient_pollution_detection.png")

model.save("earthsearch_preview_haze_model.keras")
print("Saved earthsearch_preview_haze_model.keras")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title('Accuracy')
plt.tight_layout()
plt.savefig("training_history_lenient.png", dpi=150, bbox_inches='tight')
print("Saved training_history_lenient.png")
