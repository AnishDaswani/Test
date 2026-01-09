import os
import json
import urllib.request
import urllib.error
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

IMG_SIZE = (96, 96)

def stac_search_paginated(api_url, collections, bbox, datetime, page_limit=100, max_pages=10):
    """STAC search with pagination."""
    features = []
    body = {
        "collections": collections,
        "bbox": bbox,
        "datetime": datetime,
        "limit": int(page_limit)
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type":"application/json"})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = "<no body>"
        raise RuntimeError(f"STAC search HTTP {e.code}: {e.reason}\nServer says: {err_body}")

    features.extend(page.get("features", []))

    next_href = None
    for link in page.get("links", []):
        if link.get("rel") == "next":
            next_href = link.get("href"); break

    pages_fetched = 1
    while next_href and pages_fetched < max_pages:
        try:
            with urllib.request.urlopen(next_href, timeout=30) as resp:
                page = json.loads(resp.read().decode("utf-8"))
                features.extend(page.get("features", []))
                next_href = None
                for link in page.get("links", []):
                    if link.get("rel") == "next":
                        next_href = link.get("href"); break
                pages_fetched += 1
        except Exception:
            break

    return features

def collect_preview_assets(feature):
    """Return PNG/JPEG preview asset URLs from a STAC feature."""
    urls = []
    assets = feature.get("assets", {})
    for key, meta in assets.items():
        href = meta.get("href", "")
        typ  = meta.get("type", "")
        roles = meta.get("roles", []) or []
        is_image = isinstance(typ, str) and (typ.startswith("image/png") or typ.startswith("image/jpeg"))
        likely_preview = any(r in roles for r in ["thumbnail","overview","visual","quicklook","browse"]) \
                         or key.lower() in ("thumbnail","overview","visual","quicklook","browse")
        if is_image and likely_preview and href.startswith("http"):
            urls.append(href)
    return urls

def fetch_bytes(url, max_retries=2):
    """Fetch image bytes from URL with retry logic."""
    for attempt in range(max_retries+1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
                if len(data) < 4:
                    raise ValueError("Image data too short")
                
                is_jpeg = data[:3] == b'\xff\xd8\xff'
                is_png = data[:4] == b'\x89PNG'
                is_gif = data[:6] in [b'GIF87a', b'GIF89a']
                is_bmp = data[:2] == b'BM'
                is_jp2 = (
                    (len(data) >= 12 and data[4:8] == b'jP  ') or
                    (len(data) >= 12 and data[:4] == b'\x00\x00\x00\x0c' and data[4:8] == b'jP  ')
                )
                
                if is_jp2:
                    return data

                if is_jpeg or is_png or is_gif or is_bmp:
                    return data
                else:
                    logger.warning("fetch_bytes: unsupported image format at URL: %s", url)
                    return None
        except Exception as e:
            logger.debug("fetch_bytes attempt %s failed for %s: %s", attempt, url, str(e))
            if attempt < max_retries:
                import time
                time.sleep(0.75)
            else:
                return None
    return None


def detect_image_format(img_bytes: bytes) -> str | None:
    """Return image format string if bytes match a supported image format.

    Supported formats: jpeg, png, gif, bmp, webp, jp2
    Returns None when format is unknown.
    """
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
    # WebP: 'RIFF'....'WEBP'
    if len(img_bytes) >= 12 and img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        return 'webp'
    # JPEG2000 common signatures
    if (len(img_bytes) >= 12 and img_bytes[4:8] == b'jP  ') or (len(img_bytes) >= 12 and img_bytes[:4] == b'\x00\x00\x00\x0c' and img_bytes[4:8] == b'jP  '):
        return 'jp2'

    return None

def decode_image_to_uint8(img_bytes):
    """Decode image bytes to uint8 array with robust error handling."""
    fmt = detect_image_format(img_bytes)
    if fmt is None:
        logger.debug("Unknown image format based on magic bytes; rejecting.")
        return None

    try:
        import tensorflow as tf
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        return tf.image.resize(img, IMG_SIZE, method="bilinear").numpy().astype(np.uint8)
    except Exception as e:
        logger.debug("tf.decode_image failed: %s", str(e))
    
    try:
        import tensorflow as tf
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method='INTEGER_FAST')
        return tf.image.resize(img, IMG_SIZE, method="bilinear").numpy().astype(np.uint8)
    except Exception as e:
        logger.debug("tf.decode_jpeg failed: %s", str(e))
    
    try:
        import tensorflow as tf
        img = tf.io.decode_png(img_bytes, channels=3)
        return tf.image.resize(img, IMG_SIZE, method="bilinear").numpy().astype(np.uint8)
    except Exception as e:
        logger.debug("tf.decode_png failed: %s", str(e))
    
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        return img_array
    except Exception as e:
        logger.debug("PIL decode failed: %s", str(e))
    
    return None

def build_preview_dataset(collections, bbox, datetime_range, page_limit=100):
    """Build dataset from STAC previews."""
    API_SEARCH = "https://earth-search.aws.element84.com/v1/search"
    features = stac_search_paginated(API_SEARCH, collections, bbox, datetime_range, page_limit=page_limit, max_pages=10)
    imgs = []
    skipped_count = 0
    skipped_urls = []
    for feat in features:
        urls = collect_preview_assets(feat)
        random.shuffle(urls)
        for u in urls[:1]:
            b = fetch_bytes(u)
            if b is None:
                skipped_count += 1
                skipped_urls.append((u, 'fetch_failed_or_unsupported_format'))
                continue
            arr = decode_image_to_uint8(b)
            if arr is None:
                skipped_count += 1
                skipped_urls.append((u, 'decode_failed_unknown_format'))
                continue
            imgs.append(arr)
    
    if len(imgs) == 0:
        raise RuntimeError("No decodable image preview assets found in these collections.")
    
    if skipped_count > 0:
        print(f"Note: Skipped {skipped_count} unsupported image format(s) (e.g., JPEG2000)")
    
    X = np.stack(imgs, axis=0)
    # Persist skipped URLs for later review
    try:
        import pathlib
        out_dir = os.path.join(os.getcwd(), 'plots')
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(out_dir, 'skipped_assets.log')
        if skipped_urls:
            with open(log_path, 'a', encoding='utf-8') as fh:
                for u, reason in skipped_urls:
                    fh.write(f"{u}\t{reason}\n")
            logger.warning("Wrote %d skipped asset records to %s", len(skipped_urls), log_path)
    except Exception as e:
        logger.debug("Failed to write skipped_assets.log: %s", str(e))

    return X

def haze_proxy_labels(X_uint8, percentile_threshold=50.0):
    """
    Generate haze/pollution proxy labels using edge detection and image analysis.
    
    Uses Sobel edge detection to measure image clarity. Images with lower edge
    density (more uniform/hazy) are labeled as pollution-like.
    
    Args:
        X_uint8: Numpy array of images in uint8 format (N, H, W, 3)
        percentile_threshold: Percentile threshold for labeling (default: 50.0)
        
    Returns:
        y: Binary labels (0=clear, 1=pollution_like)
        density: Edge density scores for each image
        threshold: Threshold value used for labeling
    """
    import tensorflow as tf
    
    # Normalize to [0, 1] range
    x = tf.convert_to_tensor(X_uint8, dtype=tf.float32) / 255.0
    
    # Convert to grayscale for edge detection
    gray = tf.image.rgb_to_grayscale(x)
    
    # Apply Sobel edge detection to detect image clarity
    sob = tf.image.sobel_edges(gray)
    gx, gy = sob[..., 0], sob[..., 1]
    
    # Calculate edge magnitude (clarity metric)
    mag = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-8)  # Add epsilon for stability
    
    # Average edge density per image (higher = clearer)
    density = tf.reduce_mean(mag, axis=[1, 2, 3]).numpy()
    
    # Use percentile threshold for more robust labeling
    threshold = np.percentile(density, percentile_threshold)
    
    # Label images with lower edge density as pollution-like
    y = (density < threshold).astype(np.int64)
    
    logger.info(f"Generated labels: {np.sum(y)} pollution-like ({100*np.sum(y)/len(y):.1f}%), "
                f"{len(y)-np.sum(y)} clear ({100*(len(y)-np.sum(y))/len(y):.1f}%)")
    
    return y, density, threshold

def build_model(input_shape=(96, 96, 3), num_classes=2, learning_rate=0.001, 
                optimizer='adam', dropout_rate=0.2):
    """
    Build optimized CNN model for pollution detection with customizable parameters.
    
    Architecture:
    - Data augmentation layer for improved generalization
    - 3-layer convolutional block with batch normalization
    - Global average pooling for parameter efficiency
    - Dense layers with dropout for regularization
    
    Args:
        input_shape: Shape of input images (H, W, C)
        num_classes: Number of output classes (default: 2)
        learning_rate: Initial learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        dropout_rate: Dropout rate for regularization (0.0-1.0)
        
    Returns:
        Compiled Keras model ready for training
    """
    import tensorflow as tf

    # Data augmentation for improved generalization
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomBrightness(0.1),
    ], name="augmentation")

    # Build optimized CNN architecture
    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    
    # Apply augmentation
    x = data_augmentation(inputs)
    
    # Normalize pixel values
    x = tf.keras.layers.Rescaling(1./255)(x)
    
    # First convolutional block
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1_1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1_2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool1')(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.5, name='dropout1')(x)
    
    # Second convolutional block
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2_1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2_2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool2')(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.5, name='dropout2')(x)
    
    # Third convolutional block
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3_1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3_2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool3')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout3')(x)
    
    # Global average pooling (more parameter-efficient than flatten + dense)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Classification head
    x = tf.keras.layers.Dense(128, activation='relu', name='fc1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_fc')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout_fc')(x)
    x = tf.keras.layers.Dense(64, activation='relu', name='fc2')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='pollution_detector_cnn')
    
    # Configure optimizer with learning rate scheduling support
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    else:
        logger.warning(f"Unknown optimizer '{optimizer}', defaulting to Adam")
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model with appropriate loss and metrics
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    logger.info(f"Built model with {model.count_params():,} parameters")
    return model

def make_tf_ds(x, y, batch=64, shuffle=True, buffer_size=None):
    """
    Create optimized TensorFlow dataset with prefetching and caching.
    
    Args:
        x: Input features (numpy array)
        y: Labels (numpy array)
        batch: Batch size for training
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling (default: len(x) or 1000, whichever is smaller)
        
    Returns:
        Optimized tf.data.Dataset ready for training
    """
    import tensorflow as tf
    
    # Validate inputs
    if len(x) != len(y):
        raise ValueError(f"Input and label arrays must have same length: {len(x)} != {len(y)}")
    
    if buffer_size is None:
        buffer_size = min(len(x), 1000) if shuffle else None
    
    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    
    # Shuffle with appropriate buffer size
    if shuffle and buffer_size:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42, reshuffle_each_iteration=True)
    
    # Batch with drop remainder for consistent batch sizes (optional for training)
    ds = ds.batch(batch, drop_remainder=False)
    
    # Prefetch for optimal performance
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Cache for repeated iterations (useful for validation/test sets)
    if not shuffle:
        ds = ds.cache()
    
    logger.debug(f"Created dataset: {len(x)} samples, batch_size={batch}, shuffle={shuffle}")
    
    return ds
