import os
import numpy as np
from pathlib import Path

from ml_app import training_utils

MODEL_FILE = 'earthsearch_preview_haze_model.keras'
COLLECTIONS = ['sentinel-2-l2a']
BBOX = [-84.6, 33.7, -84.2, 34.1]
DATE_RANGE = '2024-06-01T00:00:00Z/2024-12-01T23:59:59Z'
PAGE_LIMIT = 2


def load_or_build_model():
    try:
        import tensorflow as tf
    except Exception as e:
        print('TensorFlow not available:', e)
        raise

    if os.path.exists(MODEL_FILE):
        try:
            m = tf.keras.models.load_model(MODEL_FILE)
            print('Loaded model from', MODEL_FILE)
            return m
        except Exception as e:
            print('Failed to load model file, will build a fresh model:', e)

    print('Building model from code...')
    m = training_utils.build_model(input_shape=(*training_utils.IMG_SIZE,3))
    print('Model built.')
    return m


def main():
    model = load_or_build_model()

    # Fetch small preview dataset (may use network)
    try:
        print('Building preview dataset (this may use network)...')
        X = training_utils.build_preview_dataset(COLLECTIONS, bbox=BBOX, datetime_range=DATE_RANGE, page_limit=PAGE_LIMIT)
        print('Fetched dataset shape:', X.shape)
    except Exception as e:
        print('Failed to build preview dataset:', e)
        print('Using random data for quick evaluation.')
        X = np.random.randint(0, 255, size=(8, *training_utils.IMG_SIZE, 3), dtype=np.uint8)

    try:
        y, density, thresh = training_utils.haze_proxy_labels(X)
        print('Proxy labels distribution:', np.bincount(y))
    except Exception as e:
        print('Failed to compute haze proxy labels:', e)
        y = np.zeros(len(X), dtype=np.int64)

    # Predict
    try:
        preds = model.predict(X.astype('float32')/255.0, batch_size=8)
        pred_class = np.argmax(preds, axis=1)
        acc = float((pred_class == y).mean())
        print('Pred shape:', preds.shape)
        print('Accuracy vs proxy labels:', acc)

        # confusion
        cm = np.zeros((2,2), dtype=int)
        for gt, pr in zip(y, pred_class):
            if gt in (0,1) and pr in (0,1):
                cm[gt, pr] += 1
        print('Confusion matrix (rows=true, cols=pred):')
        print(cm)

    except Exception as e:
        print('Prediction failed:', e)


if __name__ == '__main__':
    main()
