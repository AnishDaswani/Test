from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os
import threading
import time
from .models import Dataset, TrainingJob, CustomImage

def index(request):
    return render(request, 'index.html')

def training_dashboard(request):
    datasets = Dataset.objects.all().order_by('-created_at')
    training_jobs = TrainingJob.objects.all().order_by('-created_at')[:10]
    return render(request, 'train.html', {
        'datasets': datasets,
        'training_jobs': training_jobs
    })

@csrf_exempt
def create_dataset(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        dataset = Dataset.objects.create(
            name=data['name'],
            description=data.get('description', ''),
            bbox=data['bbox'],
            date_range=data['date_range'],
            collections=data['collections']
        )
        threading.Thread(target=load_dataset_data, args=(dataset.id,)).start()
        return JsonResponse({'success': True, 'dataset_id': dataset.id})
    return JsonResponse({'error': 'Invalid method'}, status=400)

def load_dataset_data(dataset_id):
    try:
        import numpy as np
        from src.pollution_detector import load_previews, pollution_proxy_labels
        dataset = Dataset.objects.get(id=dataset_id)
        dataset.status = 'loading'
        dataset.save()
        
        try:
            X_all = load_previews(dataset.collections, dataset.bbox, dataset.date_range)
        except Exception as e:
            X_all = load_previews(["naip"], dataset.bbox, dataset.date_range)
            
        dataset.image_count = len(X_all)
        dataset.status = 'ready'
        dataset.save()
        
        data_path = os.path.join(settings.MEDIA_ROOT, f'datasets/{dataset_id}')
        os.makedirs(data_path, exist_ok=True)
        np.save(os.path.join(data_path, 'images.npy'), X_all)
        y_all, scores = pollution_proxy_labels(X_all, 40)
        np.save(os.path.join(data_path, 'labels.npy'), y_all)
        np.save(os.path.join(data_path, 'scores.npy'), scores)
    except Exception as e:
        dataset.status = 'error'
        dataset.save()

@csrf_exempt
def start_training(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        dataset = get_object_or_404(Dataset, id=data['dataset_id'])
        job = TrainingJob.objects.create(
            name=data['name'],
            dataset=dataset,
            epochs=data.get('epochs', 100),
            batch_size=data.get('batch_size', 32),
            learning_rate=data.get('learning_rate', 0.0005),
            config=data.get('config', {})
        )
        threading.Thread(target=train_model_background, args=(job.id,)).start()
        return JsonResponse({'success': True, 'job_id': job.id})
    return JsonResponse({'error': 'Invalid method'}, status=400)

def train_model_background(job_id):
    try:
        import numpy as np
        import tensorflow as tf
        from src.pollution_detector import build_model, train_model, evaluate_model
        job = TrainingJob.objects.get(id=job_id)
        job.status = 'training'
        job.save()
        
        data_path = os.path.join(settings.MEDIA_ROOT, f'datasets/{job.dataset.id}')
        X_all = np.load(os.path.join(data_path, 'images.npy'))
        y_all = np.load(os.path.join(data_path, 'labels.npy'))
        
        idx = np.random.permutation(len(X_all))
        n = len(idx)
        tr, va = int(0.7*n), int(0.85*n)
        x_train, y_train = X_all[idx[:tr]], y_all[idx[:tr]]
        x_val, y_val = X_all[idx[tr:va]], y_all[idx[tr:va]]
        x_test, y_test = X_all[idx[va:]], y_all[idx[va:]]
        
        model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(job.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        history = train_model(model, x_train, y_train, x_val, y_val, job.epochs)
        results = evaluate_model(model, x_test, y_test)
        
        model_path = f'models/training_job_{job.id}.keras'
        full_model_path = os.path.join(settings.MEDIA_ROOT, model_path)
        os.makedirs(os.path.dirname(full_model_path), exist_ok=True)
        model.save(full_model_path)
        
        # Manual TFJS Export logic to avoid subprocess/pickle errors
        tfjs_path = os.path.join(settings.STATIC_ROOT, f'models/job_{job.id}')
        os.makedirs(tfjs_path, exist_ok=True)
        model_config = json.loads(model.to_json())
        weights_data = []
        weight_entries = []
        for weight in model.weights:
            w_data = weight.numpy().astype('float32')
            weight_entries.append({"name": weight.name, "shape": list(w_data.shape), "dtype": "float32"})
            weights_data.append(w_data.tobytes())
        with open(os.path.join(tfjs_path, "group1-shard1of1.bin"), 'wb') as f:
            for data in weights_data: f.write(data)
        with open(os.path.join(tfjs_path, 'model.json'), 'w') as f:
            json.dump({"modelTopology": model_config, "format": "layers-model", "generatedBy": "Manual-Export", "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": weight_entries}]}, f)

        job.status = 'completed'
        job.model_path = model_path
        job.metrics = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
        }
        job.save()
    except Exception as e:
        job.status = 'failed'
        job.metrics = {'error': str(e)}
        job.save()

def get_training_status(request, job_id):
    job = get_object_or_404(TrainingJob, id=job_id)
    return JsonResponse({'status': job.status, 'metrics': job.metrics})

@csrf_exempt
def upload_custom_images(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset_id')
        dataset = get_object_or_404(Dataset, id=dataset_id)
        images = request.FILES.getlist('images')
        labels = request.POST.getlist('labels')
        count = 0
        for i, image in enumerate(images):
            if i < len(labels):
                CustomImage.objects.create(dataset=dataset, image=image, label=labels[i])
                count += 1
        return JsonResponse({'success': True, 'uploaded': count})
    return JsonResponse({'error': 'Invalid method'}, status=400)

def gradcam_view(request):
    return render(request, 'gradcam.html')
