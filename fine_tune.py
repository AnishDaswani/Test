import os
import numpy as np
from ml_app import training_utils

MODEL_IN = 'earthsearch_preview_haze_model.keras'
MODEL_OUT = 'earthsearch_preview_haze_model_finetuned.keras'

BBOX = [-84.6, 33.7, -84.2, 34.1]
DATE_RANGE = '2024-06-01T00:00:00Z/2024-12-01T23:59:59Z'


def load_model():
    import tensorflow as tf
    if os.path.exists(MODEL_IN):
        try:
            m = tf.keras.models.load_model(MODEL_IN)
            print('Loaded model', MODEL_IN)
            return m
        except Exception as e:
            print('Failed to load model, building fresh one:', e)
    return training_utils.build_model(input_shape=(*training_utils.IMG_SIZE,3))


def build_balanced_subset(X, y, max_per_class=40):
    idxs = []
    for cls in [0,1]:
        cls_idx = np.where(y==cls)[0]
        if len(cls_idx) == 0:
            continue
        np.random.shuffle(cls_idx)
        take = min(len(cls_idx), max_per_class)
        idxs.extend(cls_idx[:take].tolist())
    np.random.shuffle(idxs)
    return X[idxs], y[idxs]


def main():
    model = load_model()
    try:
        X = training_utils.build_preview_dataset(['sentinel-2-l2a'], bbox=BBOX, datetime_range=DATE_RANGE, page_limit=3)
        y, density, thresh = training_utils.haze_proxy_labels(X)
        print('Fetched preview dataset', X.shape, 'labels', np.bincount(y))
    except Exception as e:
        print('Preview fetch failed, using random data:', e)
        X = np.random.randint(0,255,(80, *training_utils.IMG_SIZE, 3), dtype=np.uint8)
        y = np.random.randint(0,2,size=(80,))

    X_sub, y_sub = build_balanced_subset(X, y, max_per_class=40)
    if len(X_sub) < 8:
        print('Not enough samples to fine-tune, aborting')
        return

    # split
    n = len(X_sub)
    tr = int(0.8 * n)
    x_tr, y_tr = X_sub[:tr], y_sub[:tr]
    x_val, y_val = X_sub[tr:], y_sub[tr:]

    import tensorflow as tf
    # recompile with lower lr
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print('Starting short fine-tune on', len(x_tr), 'samples')
    model.fit(x_tr.astype('float32')/255.0, y_tr, validation_data=(x_val.astype('float32')/255.0, y_val),
              epochs=6, batch_size=8, verbose=2)

    preds = model.predict(X.astype('float32')/255.0)
    cls = np.argmax(preds, axis=1)
    acc = (cls == y).mean()
    print('Post-finetune accuracy on preview set:', acc)

    model.save(MODEL_OUT)
    print('Saved fine-tuned model to', MODEL_OUT)

if __name__ == '__main__':
    main()
