# Earth Search STAC -> Ultra-Sensitive Pollution Detection
# Requirements: tensorflow, numpy, matplotlib

import os, json, urllib.request, random, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
API_SEARCH = "https://earth-search.aws.element84.com/v1/search"
PRIMARY_COLLECTIONS = ["sentinel-2-l2a"]
FALLBACK_COLLECTIONS = ["naip"]

BBOX = [-84.6, 33.7, -84.2, 34.1]
DATE_RANGE = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100

# 🔥 LENIENT SETTINGS - More likely to detect pollution
POLLUTION_LABEL_THRESHOLD = 40  # Label top 60% as polluted (was 40%)
PREDICTION_THRESHOLD = 0.35     # Only need 35% confidence (was 50%)

# ---------------- STAC FETCH ----------------
def stac_search(collections):
    body = json.dumps({
        "collections": collections,
        "bbox": BBOX,
        "datetime": DATE_RANGE,
        "limit": 100
    }).encode("utf-8")

    req = urllib.request.Request(API_SEARCH, body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["features"]

def decode_image(b):
    # Use the repo's magic-byte detector to avoid sending unknown bytes to TF
    try:
        from ml_app.training_utils import detect_image_format
    except Exception:
        detect_image_format = None

    if detect_image_format is not None:
        fmt = detect_image_format(b)
        if fmt is None:
            raise ValueError("Unknown image file format. One of JPEG, PNG, GIF, BMP, WebP required.")
    else:
        fmt = None

    # JPEG2000 (jp2) support via PIL; other supported types use TF's decoder
    if fmt == 'jp2':
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(b))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)

    # Fallback to TensorFlow's generic decoder for common formats
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return img.numpy().astype(np.uint8)

def load_previews(collections):
    feats = stac_search(collections)
    imgs = []
    for f in feats:
        for a in f.get("assets", {}).values():
            if "image" in str(a.get("type","")):
                try:
                    with urllib.request.urlopen(a["href"], timeout=20) as r:
                        imgs.append(decode_image(r.read()))
                    break
                except:
                    pass
    if not imgs:
        raise RuntimeError("No preview images found.")
    return np.stack(imgs)

try:
    X_all = load_previews(PRIMARY_COLLECTIONS)
except:
    X_all = load_previews(FALLBACK_COLLECTIONS)

# ---------------- AGGRESSIVE POLLUTION DETECTION ----------------
def pollution_proxy_labels(X, percentile_thresh=40):
    """
    More aggressive pollution detection:
    - Any haze, smog, or atmospheric interference
    - Reduced visibility
    - Color shifts
    - Low contrast
    """
    x = tf.cast(X, tf.float32) / 255.0
    
    # 1. Enhanced color analysis - pollution in any form
    r, g, b = x[...,0], x[...,1], x[...,2]
    
    # Detect any atmospheric interference (brown, gray, white haze)
    # High R+G relative to B OR overall grayness
    brown_gray_score = tf.reduce_mean((r + g) / (b + 0.05), axis=[1,2])
    overall_grayness = 1.0 - tf.math.reduce_std(x, axis=[1,2,3])
    color_score = brown_gray_score + overall_grayness * 2
    
    # 2. Aggressive saturation check - any desaturation is suspicious
    max_rgb = tf.reduce_max(x, axis=-1)
    min_rgb = tf.reduce_min(x, axis=-1)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    low_sat_score = (1.0 - tf.reduce_mean(saturation, axis=[1,2])) * 3
    
    # 3. Contrast - heavily weight low contrast as pollution
    gray = tf.image.rgb_to_grayscale(x)
    std_dev = tf.math.reduce_std(gray, axis=[1,2,3])
    low_contrast_score = 2.0 / (std_dev + 0.05)
    
    # 4. Haze brightness - any diffuse brightness
    mean_brightness = tf.reduce_mean(gray, axis=[1,2,3])
    haze_score = mean_brightness * low_contrast_score
    
    # 5. Color uniformity - polluted skies are more uniform
    color_std = tf.math.reduce_std(x, axis=[1,2])
    uniformity_score = 1.0 / (tf.reduce_mean(color_std, axis=-1) + 0.05)
    
    # Aggressive combination - weighted toward detecting pollution
    pollution_score = (
        0.25 * color_score +
        0.25 * low_sat_score +
        0.20 * low_contrast_score +
        0.15 * haze_score +
        0.15 * uniformity_score
    ).numpy()
    
    # Lower threshold = more images labeled as polluted
    thresh = np.percentile(pollution_score, percentile_thresh)
    y = (pollution_score > thresh).astype(np.int64)
    
    return y, pollution_score

y_all, scores = pollution_proxy_labels(X_all, POLLUTION_LABEL_THRESHOLD)

print(f"Loaded {len(X_all)} images")
print(f"Pollution labels: {np.sum(y_all)} polluted ({100*np.sum(y_all)/len(y_all):.1f}%), {len(y_all)-np.sum(y_all)} clean")

# ---------------- SPLIT ----------------
idx = np.random.permutation(len(X_all))
n = len(idx)
tr, va = int(0.7*n), int(0.85*n)

x_train, y_train = X_all[idx[:tr]], y_all[idx[:tr]]
x_val, y_val     = X_all[idx[tr:va]], y_all[idx[tr:va]]
x_test, y_test   = X_all[idx[va:]], y_all[idx[va:]]

# ---------------- MODEL WITH POLLUTION BIAS ----------------
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.2),
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMG_SIZE,3)),
    augment,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation="softmax")
])

# Use class weights to bias toward detecting pollution
class_weight = {0: 0.7, 1: 1.3}  # Bias toward pollution class

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# ---------------- TRAIN ----------------
early = tf.keras.callbacks.EarlyStopping(
    patience=10, 
    restore_best_weights=True, 
    monitor="val_loss"
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

history = model.fit(
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(512).batch(BATCH_SIZE),
    validation_data=tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=[early, reduce_lr],
    class_weight=class_weight,  # Bias toward pollution
    verbose=2
)

# ---------------- LENIENT EVALUATION ----------------
probs = model.predict(x_test, batch_size=32)

# Use lower threshold - easier to trigger pollution
y_pred = (probs[:, 1] > PREDICTION_THRESHOLD).astype(int)

acc = np.mean(y_pred == y_test)
print(f"\n{'='*50}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Pollution Detection Rate: {np.sum(y_pred) / len(y_pred):.2%}")
print(f"True Pollution Rate: {np.sum(y_test) / len(y_test):.2%}")
print(f"Threshold used: {PREDICTION_THRESHOLD:.2%} confidence")
print(f"{'='*50}")

# Show confidence distribution
print(f"\nConfidence Distribution:")
print(f"  Min pollution confidence: {probs[:, 1].min():.3f}")
print(f"  Max pollution confidence: {probs[:, 1].max():.3f}")
print(f"  Mean pollution confidence: {probs[:, 1].mean():.3f}")
print(f"  Images with >35% pollution confidence: {np.sum(probs[:, 1] > 0.35)}/{len(probs)}")
print(f"  Images with >25% pollution confidence: {np.sum(probs[:, 1] > 0.25)}/{len(probs)}")

# Confusion matrix
print("\nConfusion Matrix:")
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))
tp = np.sum((y_test == 1) & (y_pred == 1))
print(f"              Predicted Clean  Predicted Polluted")
print(f"True Clean         {tn:3d}              {fp:3d}")
print(f"True Polluted      {fn:3d}              {tp:3d}")

if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"\nPrecision (of detected pollution): {precision:.2%}")
if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"Recall (pollution detection rate): {recall:.2%}")

# ---------------- VISUALIZE SAMPLES ----------------
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle("Sample Predictions with Confidence Scores", fontsize=14)

# Show borderline cases
borderline_idx = np.where((probs[:, 1] > 0.25) & (probs[:, 1] < 0.65))[0]
high_pollution_idx = np.where(probs[:, 1] > 0.65)[0]
clean_idx = np.where(probs[:, 1] < 0.25)[0]

for i in range(5):
    if i < len(clean_idx):
        idx = clean_idx[i]
        axes[0, i].imshow(x_test[idx])
        axes[0, i].set_title(f"Clean: {probs[idx][1]:.1%}", fontsize=10)
        axes[0, i].axis('off')
    
    if i < len(borderline_idx):
        idx = borderline_idx[i]
        axes[1, i].imshow(x_test[idx])
        axes[1, i].set_title(f"Borderline: {probs[idx][1]:.1%}", fontsize=10, color='orange')
        axes[1, i].axis('off')
    
    if i < len(high_pollution_idx):
        idx = high_pollution_idx[i]
        axes[2, i].imshow(x_test[idx])
        axes[2, i].set_title(f"Polluted: {probs[idx][1]:.1%}", fontsize=10, color='red')
        axes[2, i].axis('off')

plt.tight_layout()
plt.savefig("lenient_pollution_detection.png", dpi=150, bbox_inches='tight')
print("\nSample predictions saved to 'lenient_pollution_detection.png'")

# ---------------- SAVE ----------------
model.save("earthsearch_preview_haze_model.keras")
print("Model saved to 'earthsearch_preview_haze_model.keras'")

# Save training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.savefig("training_history_lenient.png", dpi=150, bbox_inches='tight')
print("Training history saved to 'training_history_lenient.png'")

print("\n" + "="*50)
print("LENIENT MODE ACTIVE")
print(f"Threshold: {PREDICTION_THRESHOLD:.0%} confidence needed for pollution")
print(f"Your image at 55% clean = 45% polluted")
print(f"Would be classified as: POLLUTED ✓")
print("="*50)