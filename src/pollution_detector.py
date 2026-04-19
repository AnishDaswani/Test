import os
import json
import urllib.request
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100
POLLUTION_LABEL_THRESHOLD = 40
PREDICTION_THRESHOLD = 0.35

API_SEARCH = "https://earth-search.aws.element84.com/v1/search"
PRIMARY_COLLECTIONS = ["sentinel-2-l2a"]
FALLBACK_COLLECTIONS = ["naip"]

def stac_search(collections, bbox, date_range, limit=100):
    body = json.dumps({
        "collections": collections,
        "bbox": bbox,
        "datetime": date_range,
        "limit": limit
    }).encode("utf-8")

    req = urllib.request.Request(API_SEARCH, body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["features"]

def detect_image_format(b):
    if b.startswith(b'\xff\xd8'):
        return 'jpeg'
    elif b.startswith(b'\x89PNG'):
        return 'png'
    elif b.startswith(b'GIF8'):
        return 'gif'
    elif b.startswith(b'BM'):
        return 'bmp'
    elif b.startswith(b'RIFF') and b[8:12] == b'WEBP':
        return 'webp'
    elif b.startswith(b'II*\x00') or b.startswith(b'MM\x00*'):
        return 'tiff'
    elif b.startswith(b'\x00\x00\x00\x0cJXL'):
        return 'jxl'
    return None

def decode_image(b):
    img = Image.open(io.BytesIO(b))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)

def load_previews(collections, bbox, date_range):
    feats = stac_search(collections, bbox, date_range)
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
 # AI -------------------------------------------------------
def pollution_proxy_labels(X, percentile_thresh=40):
    x = tf.cast(X, tf.float32) / 255.0

    r, g, b = x[...,0], x[...,1], x[...,2]

    brown_gray_score = tf.reduce_mean((r + g) / (b + 0.05), axis=[1,2])
    overall_grayness = 1.0 - tf.math.reduce_std(x, axis=[1,2,3])
    color_score = brown_gray_score + overall_grayness * 2

    max_rgb = tf.reduce_max(x, axis=-1)
    min_rgb = tf.reduce_min(x, axis=-1)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    low_sat_score = (1.0 - tf.reduce_mean(saturation, axis=[1,2])) * 3

    gray = tf.image.rgb_to_grayscale(x)
    std_dev = tf.math.reduce_std(gray, axis=[1,2,3])
    low_contrast_score = 2.0 / (std_dev + 0.05)

    mean_brightness = tf.reduce_mean(gray, axis=[1,2,3])
    haze_score = mean_brightness * low_contrast_score

    color_std = tf.math.reduce_std(x, axis=[1,2])
    uniformity_score = 1.0 / (tf.reduce_mean(color_std, axis=-1) + 0.05)

    pollution_score = (
        0.25 * color_score +
        0.25 * low_sat_score +
        0.20 * low_contrast_score +
        0.15 * haze_score +
        0.15 * uniformity_score
    ).numpy()

    thresh = np.percentile(pollution_score, percentile_thresh)
    y = (pollution_score > thresh).astype(np.int64)

    return y, pollution_score
#------------------------------------------------------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE,3)),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.2),
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    class_weight = {0: 0.7, 1: 1.3}

    early = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(512).batch(BATCH_SIZE),
        validation_data=tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE),
        epochs=epochs,
        callbacks=[early, reduce_lr],
        class_weight=class_weight,
        verbose=2
    )

    return history

def evaluate_model(model, X_test, y_test):
    probs = model.predict(X_test, batch_size=32)
    y_pred = (probs[:, 1] > PREDICTION_THRESHOLD).astype(int)

    acc = np.mean(y_pred == y_test)
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'pollution_rate': np.sum(y_pred) / len(y_pred),
        'true_pollution_rate': np.sum(y_test) / len(y_test),
        'predictions': y_pred,
        'probabilities': probs
    }

def save_training_plots(history, save_path="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.legend()
    ax1.set_title('Loss')

    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.legend()
    ax2.set_title('Accuracy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_sample_predictions(X_test, y_test, probs, save_path="sample_predictions.png"):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle("Sample Predictions with Confidence Scores", fontsize=14)

    borderline_idx = np.where((probs[:, 1] > 0.25) & (probs[:, 1] < 0.65))[0]
    high_pollution_idx = np.where(probs[:, 1] > 0.65)[0]
    clean_idx = np.where(probs[:, 1] < 0.25)[0]

    for i in range(5):
        if i < len(clean_idx):
            idx = clean_idx[i]
            axes[0, i].imshow(X_test[idx])
            axes[0, i].set_title(f"Clean: {probs[idx][1]:.1%}", fontsize=10)
            axes[0, i].axis('off')

        if i < len(borderline_idx):
            idx = borderline_idx[i]
            axes[1, i].imshow(X_test[idx])
            axes[1, i].set_title(f"Borderline: {probs[idx][1]:.1%}", fontsize=10, color='orange')
            axes[1, i].axis('off')

        if i < len(high_pollution_idx):
            idx = high_pollution_idx[i]
            axes[2, i].imshow(X_test[idx])
            axes[2, i].set_title(f"Polluted: {probs[idx][1]:.1%}", fontsize=10, color='red')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()