import os
import sys
import types
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Provide a lightweight `cv2` stub to avoid importing the full native
# OpenCV during Keras/TensorFlow import (Keras may import visualization
# which imports `cv2` if available). The real `cv2` is useful for
# overlays but its native import can hang startup; this stub implements
# the tiny subset we need: `resize`, `addWeighted`, and `imwrite`.
if 'cv2' not in sys.modules:
    try:
        from PIL import Image
        cv2_stub = types.ModuleType('cv2')
        def _resize(img, dsize):
            # dsize is (width, height)
            pil = Image.fromarray(img.astype('uint8'))
            pil = pil.resize((int(dsize[0]), int(dsize[1])))
            return np.array(pil)
        def _addWeighted(src1, alpha, src2, beta, gamma):
            out = (src1.astype('float32') * float(alpha) + src2.astype('float32') * float(beta) + float(gamma))
            out = np.clip(out, 0, 255).astype('uint8')
            return out
        def _imwrite(path, arr):
            Image.fromarray(arr.astype('uint8')).save(path)
        cv2_stub.resize = _resize
        cv2_stub.addWeighted = _addWeighted
        cv2_stub.imwrite = _imwrite
        sys.modules['cv2'] = cv2_stub
    except Exception:
        # If PIL isn't available, skip stub and allow real cv2 import to fail later
        pass

from ml_app import training_utils

MODEL_FILE = 'earthsearch_preview_haze_model.keras'
COLLECTIONS = ['sentinel-2-l2a']
BBOX = [-84.6, 33.7, -84.2, 34.1]
DATE_RANGE = '2024-06-01T00:00:00Z/2024-12-01T23:59:59Z'
PAGE_LIMIT = 3
OUT_DIR = Path('plots')
OUT_DIR.mkdir(exist_ok=True)


def load_model():
    import tensorflow as tf
    if os.path.exists(MODEL_FILE):
        m = tf.keras.models.load_model(MODEL_FILE)
        print('Loaded model', MODEL_FILE)
        return m
    print('No saved model found; building fresh model')
    return training_utils.build_model(input_shape=(*training_utils.IMG_SIZE,3))


def grad_cam(model, img_tensor, class_index):
    import tensorflow as tf
    # Simple Grad-CAM for Keras models. Find the last Conv2D layer,
    # allowing for nested `Sequential`/`Model` containers.
    img = tf.convert_to_tensor(img_tensor[None, ...], dtype=tf.float32)

    def find_last_conv(model):
        # search reversed top-level layers; be robust to differing Keras classes
        def is_conv_layer(l):
            if getattr(l, 'kernel', None) is not None:
                return True
            name = l.__class__.__name__.lower()
            return 'conv' in name

        for layer in reversed(model.layers):
            if is_conv_layer(layer):
                return layer
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                for sub in reversed(layer.layers):
                    if is_conv_layer(sub):
                        return sub
        return None

    last_conv = find_last_conv(model)
    if last_conv is None:
        raise RuntimeError('No Conv2D layer found for Grad-CAM')

    # Ensure the model has been called so tensors are connected
    _ = model.predict(img, verbose=0)

    # Build the activation tensor for the located conv layer by applying
    # layers symbolically to the model input. This avoids relying on
    # `layer.output` which may be unset on some saved models.
    x = model.inputs[0]
    conv_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer is last_conv or layer.name == last_conv.name:
            conv_tensor = x
            break
    if conv_tensor is None:
        # try searching nested containers
        for layer in model.layers:
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                for sub in layer.layers:
                    x = sub(x)
                    if sub is last_conv or sub.name == last_conv.name:
                        conv_tensor = x
                        break
                if conv_tensor is not None:
                    break
    if conv_tensor is None:
        raise RuntimeError('Could not construct conv activation tensor for Grad-CAM')

    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[conv_tensor, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    # expand to 3 channels and resize to image size using cv2 (stub or real)
    import cv2
    heatmap_rgb = np.stack([heatmap, heatmap, heatmap], axis=-1)
    heatmap_resized = cv2.resize(heatmap_rgb, (img_tensor.shape[1], img_tensor.shape[0]))
    return heatmap_resized


def save_gradcam_overlay(img, heatmap, out_path):
    import cv2
    img = img.astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(str(out_path), overlay)


def main():
    model = load_model()

    try:
        X = training_utils.build_preview_dataset(COLLECTIONS, bbox=BBOX, datetime_range=DATE_RANGE, page_limit=PAGE_LIMIT)
        print('Fetched dataset shape', X.shape)
    except Exception as e:
        print('Failed to fetch previews:', e)
        X = np.random.randint(0,255,(16, *training_utils.IMG_SIZE, 3), dtype=np.uint8)

    y, density, thresh = training_utils.haze_proxy_labels(X)
    print('Proxy label counts:', np.bincount(y))

    preds = model.predict(X.astype('float32')/255.0, batch_size=8)
    probs = preds[:,1]
    pred_class = np.argmax(preds, axis=1)

    print('Probs min/max/mean:', probs.min(), probs.max(), probs.mean())

    # Confusion
    cm = np.zeros((2,2), dtype=int)
    for gt, pr in zip(y, pred_class):
        if gt in (0,1) and pr in (0,1):
            cm[gt,pr] += 1
    print('Confusion matrix:\n', cm)

    # Threshold sweep
    print('\nThreshold sweep: [threshold, detection_rate, acc]')
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        yp = (probs > t).astype(int)
        acc = (yp == y).mean()
        det = yp.mean()
        print(f'{t:.2f}\t{det:.3f}\t{acc:.3f}')

    # Save some Grad-CAM overlays for misclassified polluted/clean
    mis_idx = [i for i,(gt,pr) in enumerate(zip(y,pred_class)) if gt!=pr]
    os.makedirs(OUT_DIR / 'gradcam', exist_ok=True)
    import cv2
    for i, idx in enumerate(mis_idx[:6]):
        img = X[idx]
        cls = pred_class[idx]
        try:
            heat = grad_cam(model, img.astype('float32')/255.0, cls)
            outp = OUT_DIR / 'gradcam' / f'mis_{i}_idx{idx}_gt{y[idx]}_pr{cls}.png'
            save_gradcam_overlay((img).astype('uint8'), heat, outp)
            print('Wrote', outp)
        except Exception as e:
            print('Grad-CAM failed for idx', idx, e)

if __name__ == '__main__':
    main()
