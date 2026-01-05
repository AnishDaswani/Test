import os
import json
import threading
import numpy as np
import tensorflow as tf
from django.conf import settings
import matplotlib
matplotlib.use('Agg')

training_state = {
    'status': 'idle',
    'current_epoch': 0,
    'total_epochs': 0,
    'history': {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    },
    'message': '',
    'progress_percent': 0,
    'final_stats': {
        'test_accuracy': None,
        'test_loss': None,
        'train_accuracy': None,
        'train_loss': None,
        'val_accuracy': None,
        'val_loss': None,
        'total_epochs_trained': 0
    }
}

def reset_training_state():
    global training_state
    training_state = {
        'status': 'idle',
        'current_epoch': 0,
        'total_epochs': 0,
        'history': {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        },
        'message': '',
        'progress_percent': 0,
        'final_stats': {
            'test_accuracy': None,
            'test_loss': None,
            'train_accuracy': None,
            'train_loss': None,
            'val_accuracy': None,
            'val_loss': None,
            'total_epochs_trained': 0
        }
    }

class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to track training progress."""
    
    def on_train_begin(self, logs=None):
        training_state['status'] = 'running'
        training_state['message'] = 'Training started...'
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        training_state['current_epoch'] = epoch + 1
        
        if 'loss' in logs:
            training_state['history']['loss'].append(float(logs['loss']))
        if 'accuracy' in logs:
            training_state['history']['accuracy'].append(float(logs['accuracy']))
        if 'val_loss' in logs:
            training_state['history']['val_loss'].append(float(logs['val_loss']))
        if 'val_accuracy' in logs:
            training_state['history']['val_accuracy'].append(float(logs['val_accuracy']))
        
        if training_state['total_epochs'] > 0:
            training_state['progress_percent'] = int((epoch + 1) / training_state['total_epochs'] * 100)
        
        epoch_msg = f"Epoch {epoch + 1}/{training_state['total_epochs']}"
        if 'loss' in logs and 'accuracy' in logs:
            training_state['message'] = f"{epoch_msg} - Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}"
            if 'val_loss' in logs and 'val_accuracy' in logs:
                training_state['message'] += f", Val Loss: {logs['val_loss']:.4f}, Val Acc: {logs['val_accuracy']:.4f}"
    
    def on_train_end(self, logs=None):
        training_state['status'] = 'completed'
        training_state['message'] = 'Training completed!'
        training_state['progress_percent'] = 100

def train_model_in_background(epochs=50, batch_size=64, img_size=(96, 96), 
                              learning_rate=0.001, optimizer='adam', dropout_rate=0.2,
                              train_split=0.7, val_split=0.15, early_stopping_patience=5,
                              page_limit=100):
    """Train the model in a background thread."""
    
    def train():
        try:
            from ml_app.views import get_paths
            from ml_app.training_utils import (
                build_preview_dataset, haze_proxy_labels, build_model, make_tf_ds
            )
            
            reset_training_state()
            training_state['total_epochs'] = epochs
            training_state['status'] = 'preparing'
            training_state['message'] = 'Loading data...'
            
            MODEL_PATH, _, PLOTS_DIR = get_paths()
            
            PRIMARY_COLLECTIONS = ["sentinel-2-l2a"]
            FALLBACK_COLLECTIONS = ["naip"]
            BBOX = [-84.6, 33.7, -84.2, 34.1]
            DATE_RANGE = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"
            
            training_state['message'] = 'Fetching satellite images...'
            try:
                X_all = build_preview_dataset(PRIMARY_COLLECTIONS, BBOX, DATE_RANGE, page_limit)
            except Exception as e:
                training_state['message'] = f'Primary collection failed, trying fallback... ({str(e)[:50]})'
                X_all = build_preview_dataset(FALLBACK_COLLECTIONS, BBOX, DATE_RANGE, page_limit)
            
            training_state['message'] = 'Generating labels...'
            y_all, _, _ = haze_proxy_labels(X_all)
            
            training_state['message'] = 'Splitting dataset...'
            rng = np.random.default_rng(42)
            idx = np.arange(len(X_all))
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(train_split * n)
            n_val = int(val_split * n)
            train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
            x_train, y_train = X_all[train_idx], y_all[train_idx]
            x_val, y_val = X_all[val_idx], y_all[val_idx]
            x_test, y_test = X_all[test_idx], y_all[test_idx]
            
            training_state['message'] = 'Building model...'
            model = build_model(
                input_shape=(img_size[0], img_size[1], 3), 
                num_classes=2,
                learning_rate=learning_rate,
                optimizer=optimizer,
                dropout_rate=dropout_rate
            )
            
            train_ds = make_tf_ds(x_train, y_train, batch=batch_size, shuffle=True)
            val_ds = make_tf_ds(x_val, y_val, batch=batch_size, shuffle=False)
            
            progress_cb = ProgressCallback()
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', patience=early_stopping_patience, restore_best_weights=True
            )
            
            training_state['message'] = 'Starting training...'
            
            history_obj = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                verbose=0,
                callbacks=[progress_cb, early_stop]
            )
            
            training_state['message'] = 'Evaluating model...'
            if 'x_test' in locals() and x_test is not None and len(x_test) > 0:
                test_ds = make_tf_ds(x_test, y_test, batch=batch_size, shuffle=False)
                test_loss, test_acc = model.evaluate(test_ds, verbose=0)
                training_state['final_stats']['test_accuracy'] = float(test_acc)
                training_state['final_stats']['test_loss'] = float(test_loss)
            else:
                val_loss, val_acc = model.evaluate(val_ds, verbose=0)
                training_state['final_stats']['test_accuracy'] = float(val_acc)
                training_state['final_stats']['test_loss'] = float(val_loss)
            
            if history_obj.history:
                train_acc_history = history_obj.history.get('accuracy', [])
                train_loss_history = history_obj.history.get('loss', [])
                val_acc_history = history_obj.history.get('val_accuracy', [])
                val_loss_history = history_obj.history.get('val_loss', [])
                
                if train_acc_history:
                    training_state['final_stats']['train_accuracy'] = float(train_acc_history[-1])
                if train_loss_history:
                    training_state['final_stats']['train_loss'] = float(train_loss_history[-1])
                if val_acc_history:
                    training_state['final_stats']['val_accuracy'] = float(val_acc_history[-1])
                if val_loss_history:
                    training_state['final_stats']['val_loss'] = float(val_loss_history[-1])
                
                training_state['final_stats']['total_epochs_trained'] = len(train_acc_history)
            
            training_state['message'] = 'Saving model...'
            model.save(MODEL_PATH)
            print(f"✓ Model saved to: {MODEL_PATH}")
            
            training_state['status'] = 'completed'
            training_state['message'] = 'Training completed successfully!'
            training_state['progress_percent'] = 100
            
        except Exception as e:
            training_state['status'] = 'error'
            training_state['message'] = f'Error: {str(e)}'
            import traceback
            training_state['error'] = traceback.format_exc()
    
    thread = threading.Thread(target=train, daemon=True)
    thread.start()
    return thread
