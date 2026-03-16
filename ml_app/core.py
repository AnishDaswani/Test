import os
import json
import random
import logging
import urllib.request
import urllib.error
import numpy as np

logger = logging.getLogger(__name__)
IMG_SIZE = (96, 96)
API_SEARCH = "https://earth-search.aws.element84.com/v1/search"


def stac_search_paginated(api_url, collections, bbox, datetime_range, page_limit=100, max_pages=10):
    body = json.dumps({"collections": collections, "bbox": bbox, "datetime": datetime_range, "limit": int(page_limit)}).encode("utf-8")
    req = urllib.request.Request(api_url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            page = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else "<no body>"
        raise RuntimeError(f"STAC HTTP {e.code}: {e.reason}\n{err_body}") from e
    features = list(page.get("features", []))
    next_href = next((link.get("href") for link in page.get("links", []) if link.get("rel") == "next"), None)
    pages = 1
    while next_href and pages < max_pages:
        try:
            with urllib.request.urlopen(next_href, timeout=30) as r:
                page = json.loads(r.read().decode("utf-8"))
                features.extend(page.get("features", []))
                next_href = next((link.get("href") for link in page.get("links", []) if link.get("rel") == "next"), None)
                pages += 1
        except Exception:
            break
    return features


def _collect_preview_urls(feat):
    urls = []
    for key, meta in (feat.get("assets") or {}).items():
        href = meta.get("href", "")
        typ = str(meta.get("type", ""))
        roles = meta.get("roles") or []
        if not (typ.startswith("image/png") or typ.startswith("image/jpeg")) or not href.startswith("http"):
            continue
        preview_roles = ("thumbnail", "overview", "visual", "quicklook", "browse")
        if any(r in roles for r in preview_roles) or key.lower() in preview_roles:
            urls.append(href)
    return urls


def detect_image_format(img_bytes):
    if not img_bytes or len(img_bytes) < 4:
        return None
    if img_bytes[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if img_bytes[:4] == b'\x89PNG':
        return 'png'
    if img_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    if img_bytes[:2] == b'BM':
        return 'bmp'
    if len(img_bytes) >= 12 and img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        return 'webp'
    if len(img_bytes) >= 12 and (img_bytes[4:8] == b'jP  ' or (img_bytes[:4] == b'\x00\x00\x00\x0c' and img_bytes[4:8] == b'jP  ')):
        return 'jp2'
    return None


def decode_image_to_uint8(img_bytes, size=None):
    size = size or IMG_SIZE
    if detect_image_format(img_bytes) is None:
        return None
    try:
        import tensorflow as tf
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        return tf.image.resize(img, size, method="bilinear").numpy().astype(np.uint8)
    except Exception:
        pass
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
    except Exception:
        return None


def _fetch_bytes(url, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
                if len(data) < 4:
                    raise ValueError("Image data too short")
                if data[:3] == b'\xff\xd8\xff' or data[:4] == b'\x89PNG':
                    return data
                if data[:6] in (b'GIF87a', b'GIF89a') or data[:2] == b'BM':
                    return data
                if len(data) >= 12 and (data[4:8] == b'jP  ' or (data[:4] == b'\x00\x00\x00\x0c' and data[4:8] == b'jP  ')):
                    return data
                logger.warning("Unsupported image format at %s", url)
                return None
        except Exception:
            if attempt < max_retries:
                import time
                time.sleep(0.75)
            else:
                return None
    return None


def build_preview_dataset(collections, bbox, datetime_range, page_limit=100, img_size=None):
    img_size = img_size or IMG_SIZE
    features = stac_search_paginated(API_SEARCH, collections, bbox, datetime_range, page_limit=page_limit, max_pages=10)
    imgs = []
    skipped = []
    for feat in features:
        urls = _collect_preview_urls(feat)
        random.shuffle(urls)
        for u in urls[:1]:
            raw = _fetch_bytes(u)
            if raw is None:
                skipped.append((u, 'fetch_failed'))
                continue
            arr = decode_image_to_uint8(raw, size=img_size)
            if arr is None:
                skipped.append((u, 'decode_failed'))
                continue
            imgs.append(arr)
    if not imgs:
        raise RuntimeError("No decodable preview assets found.")
    if skipped:
        out_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(out_dir, exist_ok=True)
        try:
            log_path = os.path.join(out_dir, 'skipped_assets.log')
            with open(log_path, 'a', encoding='utf-8') as f:
                for u, r in skipped:
                    f.write(f"{u}\t{r}\n")
        except Exception:
            pass
    return np.stack(imgs)


def haze_proxy_labels(X_uint8, percentile_threshold=50.0):
    import tensorflow as tf
    x = tf.cast(X_uint8, tf.float32) / 255.0
    gray = tf.image.rgb_to_grayscale(x)
    sob = tf.image.sobel_edges(gray)
    mag = tf.sqrt(tf.square(sob[..., 0]) + tf.square(sob[..., 1]) + 1e-8)
    edge_density = tf.reduce_mean(mag, axis=[1, 2, 3]).numpy()
    lo, hi = edge_density.min(), edge_density.max()
    clarity = (edge_density - lo) / (hi - lo + 1e-8)
    mx, mn = tf.reduce_max(x, axis=-1), tf.reduce_min(x, axis=-1)
    sat = (mx - mn) / (mx + 1e-6)
    mean_sat = tf.reduce_mean(sat, axis=[1, 2]).numpy()
    mean_bright = tf.reduce_mean(gray, axis=[1, 2, 3]).numpy()
    haze_score = (1.0 - mean_sat) * 0.6 + np.clip(mean_bright, 0, 1) * 0.4
    pollution_score = (1.0 - clarity) * 0.55 + haze_score * 0.45
    threshold = np.percentile(pollution_score, percentile_threshold)
    y = (pollution_score > threshold).astype(np.int64)
    return y, pollution_score, threshold


def focal_loss(gamma=2.0):
    import tensorflow as tf
    def _loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        depth = tf.shape(y_pred)[1]
        idx = tf.minimum(y_true, depth - 1)
        pt = tf.reduce_sum(y_pred * tf.one_hot(idx, depth), axis=1)
        pt = tf.clip_by_value(pt, 1e-7, 1.0 - 1e-7)
        return -tf.reduce_mean(tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return _loss


def _se_block(x, ratio=8):
    import tensorflow as tf
    ch = x.shape[-1]
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
    excite = tf.keras.layers.Dense(max(ch // ratio, 8), activation='relu')(squeeze)
    excite = tf.keras.layers.Dense(ch, activation='sigmoid')(excite)
    return tf.keras.layers.Multiply()([x, tf.keras.layers.Reshape((1, 1, ch))(excite)])


def build_model(input_shape=(96, 96, 3), num_classes=2, learning_rate=0.001, optimizer='adam', dropout_rate=0.25, use_focal_loss=False):
    import tensorflow as tf
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ], name="augmentation")
    inp = tf.keras.layers.Input(shape=input_shape, name='input')
    x = aug(inp)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    for i, filters in enumerate((32, 64, 128, 256)):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(dropout_rate * (0.5 if i < 2 else 1.0))(x)
    x = _se_block(x, ratio=16)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name='pollution_detector_cnn')
    optimizers = {
        'adam': lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        'rmsprop': lambda: tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9),
    }
    opt = optimizers.get(optimizer.lower(), optimizers['adam'])()
    loss_fn = focal_loss(gamma=2.0) if use_focal_loss else 'sparse_categorical_crossentropy'
    model.compile(optimizer=opt, loss=loss_fn,
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model


def make_tf_ds(x, y, batch=64, shuffle=True, buffer_size=None):
    import tensorflow as tf
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {len(x)} != {len(y)}")
    buf = min(len(x), 1000) if shuffle and buffer_size is None else buffer_size
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle and buf:
        ds = ds.shuffle(buffer_size=buf, seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    if not shuffle:
        ds = ds.cache()
    return ds


def _build_gradcam_model(model, input_shape):
    import tensorflow as tf
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No Conv2D layer found for Grad-CAM.")
    last_conv = conv_layers[-1]
    try:
        return tf.keras.Model(inputs=model.inputs, outputs=[last_conv.output, model.output])
    except Exception:
        inp = tf.keras.Input(shape=input_shape)
        x = inp
        last_out = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_out = x
        if last_out is None:
            raise ValueError("No Conv2D output for Grad-CAM.")
        return tf.keras.Model(inputs=inp, outputs=[last_out, x])


def make_gradcam_heatmap(img_array, model, img_size=(96, 96)):
    import tensorflow as tf
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    grad_model = _build_gradcam_model(model, (img_size[0], img_size[1], 3))
    batch = np.expand_dims(img_array, axis=0) if img_array.ndim == 3 else img_array
    batch_tf = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch_tf)
        pred_idx = tf.argmax(preds[0])
        class_ch = preds[:, pred_idx]
    grads = tape.gradient(class_ch, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    weighted = conv_out[0] * pooled[tf.newaxis, tf.newaxis, :]
    heat = tf.reduce_sum(weighted, axis=-1)
    heat = tf.maximum(heat, 0) / (tf.math.reduce_max(heat) + 1e-9)
    return heat.numpy()


def overlay_gradcam_on_image(img_uint8, heatmap, alpha=0.5, cmap='jet'):
    import tensorflow as tf
    from matplotlib import cm as mpl_cm
    h, w = img_uint8.shape[0], img_uint8.shape[1]
    h_resized = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy().squeeze()
    colored = (mpl_cm.get_cmap(cmap)(h_resized)[..., :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * img_uint8).astype(np.uint8)


def save_gradcam_visualization(img_uint8, heatmap, overlay, predicted_class, confidence, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_uint8)
    axes[0].set_title(f'Original\n{predicted_class} {confidence:.2%}')
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150, format='png')
    plt.close(fig)
    return save_path
