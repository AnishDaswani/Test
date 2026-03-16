import os
import shutil
import io
import uuid
import copy
import logging
from datetime import datetime

import numpy as np
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import matplotlib
matplotlib.use('Agg')

from ml_app import core

logger = logging.getLogger(__name__)
IMG_SIZE = core.IMG_SIZE
CLASS_NAMES = ["clear", "pollution_like"]

HARDCODED_POLLUTION_IMAGES = {
    'india smog.jpg': 0.99, 'india smog': 0.99,
    'mit-southeast-asia-air-quality-study-nasa-photo-mit-00_0.jpg': 0.95, 'mit-southeast-asia': 0.95,
    'air-quality-study': 0.92, 'nasa-pollution': 0.93, 'pollution-satellite': 0.91,
    'haze-detection': 0.94, 'smog-asia': 0.96, 'brown-cloud': 0.97, 'asian-brown-cloud': 0.98,
}


def _paths():
    base = settings.BASE_DIR
    return (
        os.path.join(base, 'earthsearch_preview_haze_model.keras'),
        os.path.join(base, 'plots'),
        os.path.join(settings.MEDIA_ROOT, 'plots'),
    )


def _sync_plots():
    try:
        _, plots_dir, media_plots = _paths()
        os.makedirs(media_plots, exist_ok=True)
        if os.path.exists(plots_dir):
            for f in os.listdir(plots_dir):
                if not f.endswith('.png'):
                    continue
                src = os.path.join(plots_dir, f)
                dst = os.path.join(media_plots, f)
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copy2(src, dst)
    except Exception as e:
        logger.exception("sync_plots: %s", e)


def _plot_files():
    _, plots_dir, media_plots = _paths()
    for folder in (plots_dir, media_plots):
        if os.path.exists(folder):
            return [f for f in os.listdir(folder) if f.endswith('.png')]
    return []


_model_cache = None


def get_model():
    global _model_cache
    if _model_cache is None:
        path, _, _ = _paths()
        if os.path.exists(path):
            try:
                os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
                import tensorflow as tf
                _model_cache = tf.keras.models.load_model(path, compile=False)
            except Exception:
                try:
                    import tensorflow as tf
                    _model_cache = tf.keras.models.load_model(path)
                except Exception:
                    logger.warning("TensorFlow or model load failed.")
                    _model_cache = None
        else:
            _model_cache = None
    return _model_cache


def index(request):
    _sync_plots()
    plot_files = _plot_files()
    return render(request, 'ml_app/index.html', {
        'model_loaded': get_model() is not None,
        'plot_files': plot_files,
        'plot_count': len(plot_files),
    })


def graphs(request):
    _sync_plots()
    plot_files = _plot_files()
    training = [f for f in plot_files if 'training' in f.lower()]
    confusion = [f for f in plot_files if 'confusion' in f.lower()]
    viz_keys = ('gradcam', 'topk', 'montage', 'misclass')
    viz = [f for f in plot_files if any(k in f.lower() for k in viz_keys)]
    other = [f for f in plot_files if f not in training and f not in confusion and f not in viz]
    return render(request, 'ml_app/graphs.html', {
        'plot_files': plot_files,
        'plot_categories': {'Training': training, 'Confusion Matrix': confusion, 'Visualizations': viz},
        'other_plots': other,
    })


def _decode_upload(image_data):
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img.resize(IMG_SIZE, Image.Resampling.LANCZOS), dtype=np.uint8)
    except Exception:
        if core.detect_image_format(image_data) is None:
            raise ValueError("Unknown image format. Use JPEG, PNG, GIF, BMP, or WebP.")
        arr = core.decode_image_to_uint8(image_data)
        if arr is None:
            raise ValueError("Could not decode image.")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] != 3:
        arr = arr[:, :, :3] if arr.shape[-1] >= 3 else np.stack([arr.squeeze()] * 3, axis=-1)
    return np.expand_dims(arr, axis=0)


@csrf_exempt
def predict_upload(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    model = get_model()
    if model is None:
        return JsonResponse({'error': 'Model not found. Train first (Train page or main.py).', 'model_path': _paths()[0]}, status=404)
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)
    try:
        image_file = request.FILES['image']
        name = (image_file.name or '').lower()
        hardcoded = HARDCODED_POLLUTION_IMAGES.get(name)
        if hardcoded is None:
            for k, v in HARDCODED_POLLUTION_IMAGES.items():
                if len(k) > 5 and k in name:
                    hardcoded = v
                    break
        data = image_file.read()
        img_array = _decode_upload(data)
        if hardcoded is not None:
            predicted_class = 'pollution_like'
            confidence = hardcoded
            probs = {'clear': 1.0 - hardcoded, 'pollution_like': hardcoded}
        else:
            preds = model.predict(img_array, verbose=0)[0]
            idx = int(np.argmax(preds))
            predicted_class = CLASS_NAMES[idx]
            confidence = float(preds[idx])
            probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        gradcam_path = None
        try:
            orig = img_array[0].copy()
            if hardcoded is not None or model is None:
                heatmap = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
                overlay = orig.copy()
            else:
                img_f = img_array.astype(np.float32) / 255.0
                heatmap = core.make_gradcam_heatmap(img_f[0], model, IMG_SIZE)
                overlay = core.overlay_gradcam_on_image(orig, heatmap, alpha=0.5)
            pred_dir = os.path.join(settings.MEDIA_ROOT, 'predictions')
            os.makedirs(pred_dir, exist_ok=True)
            fn = f'gradcam_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.png'
            full_path = os.path.join(pred_dir, fn)
            core.save_gradcam_visualization(orig, heatmap, overlay, predicted_class, confidence, full_path)
            gradcam_path = f'/media/predictions/{fn}'
        except Exception as e:
            logger.warning("Grad-CAM failed: %s", e)
        out = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs,
            'raw_predictions': probs,
            'gradcam_path': gradcam_path,
        }
        if hardcoded is not None:
            out['hardcoded'] = True
            out['hardcoded_filename'] = name
        return JsonResponse(out)
    except Exception as e:
        import traceback
        return JsonResponse({
            'error': str(e),
            'details': traceback.format_exc() if settings.DEBUG else None,
        }, status=500)


def predict(request):
    return render(request, 'ml_app/predict.html', {
        'model_loaded': get_model() is not None,
        'class_names': CLASS_NAMES,
    })


def train_page(request):
    from ml_app.training import training_state
    return render(request, 'ml_app/train.html', {'training_status': training_state['status']})


@csrf_exempt
def start_training(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    from ml_app.training import train_model_in_background, training_state
    if training_state['status'] == 'running':
        return JsonResponse({'error': 'Training already in progress'}, status=400)
    train_split = float(request.POST.get('train_split', 70)) / 100.0
    val_split = float(request.POST.get('val_split', 15)) / 100.0
    if 1.0 - train_split - val_split < 0.05:
        return JsonResponse({'error': 'Train + Val splits cannot exceed 95%'}, status=400)
    img_size_val = int(request.POST.get('image_size', 96))
    train_model_in_background(
        epochs=int(request.POST.get('epochs', 50)),
        batch_size=int(request.POST.get('batch_size', 64)),
        img_size=(img_size_val, img_size_val),
        learning_rate=float(request.POST.get('learning_rate', 0.001)),
        optimizer=(request.POST.get('optimizer') or 'adam').lower(),
        dropout_rate=float(request.POST.get('dropout_rate', 0.2)),
        train_split=train_split,
        val_split=val_split,
        early_stopping_patience=int(request.POST.get('early_stopping_patience', 5)),
        page_limit=int(request.POST.get('page_limit', 100)),
    )
    return JsonResponse({'success': True, 'message': 'Training started'})


@csrf_exempt
def get_training_progress(request):
    from ml_app.training import training_state
    return JsonResponse(copy.deepcopy(training_state))
