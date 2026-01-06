import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
    # Simple Grad-CAM for Keras Sequential conv models
    img = tf.convert_to_tensor(img_tensor[None,...], dtype=tf.float32)
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if hasattr(layer, 'output') and len(getattr(layer, 'output').shape) == 4:
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        raise RuntimeError('No conv layer found for Grad-CAM')
    last_conv = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model([
        model.inputs], [last_conv.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = np.repeat(heatmap, 3, axis=-1)
    heatmap = np.array(plt.imresize(heatmap, (img_tensor.shape[0], img_tensor.shape[1]))) if hasattr(plt, 'imresize') else np.array(plt.imshow(heatmap, interpolation='bilinear').get_array())
    return heatmap


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
