import os
import threading
import logging
import numpy as np
import tensorflow as tf
from django.conf import settings

import matplotlib
matplotlib.use('Agg')

from ml_app import core
from ml_app.views import _paths

logger = logging.getLogger(__name__)

training_state = {
    'status': 'idle',
    'current_epoch': 0,
    'total_epochs': 0,
    'history': {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []},
    'message': '',
    'progress_percent': 0,
    'final_stats': {
        'test_accuracy': None,
        'test_loss': None,
        'train_accuracy': None,
        'train_loss': None,
        'val_accuracy': None,
        'val_loss': None,
        'total_epochs_trained': 0,
    },
}


def _reset_state():
    global training_state
    final_stats = {k: None for k in training_state['final_stats']}
    final_stats['total_epochs_trained'] = 0
    training_state = {
        'status': 'idle',
        'current_epoch': 0,
        'total_epochs': 0,
        'history': {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []},
        'message': '',
        'progress_percent': 0,
        'final_stats': final_stats,
    }


class ProgressCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        training_state['status'] = 'running'
        training_state['message'] = 'Training started...'

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        training_state['current_epoch'] = epoch + 1
        for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
            if k in logs:
                training_state['history'][k].append(float(logs[k]))
        total = training_state['total_epochs']
        if total > 0:
            training_state['progress_percent'] = int((epoch + 1) / total * 100)
        training_state['message'] = (
            f"Epoch {epoch + 1}/{total} - "
            f"Loss: {logs.get('loss', 0):.4f}, Acc: {logs.get('accuracy', 0):.4f}"
        )

    def on_train_end(self, logs=None):
        training_state['status'] = 'completed'
        training_state['message'] = 'Training completed!'
        training_state['progress_percent'] = 100


def train_model_in_background(epochs=50, batch_size=64, img_size=(96, 96), learning_rate=0.001, optimizer='adam',
                              dropout_rate=0.2, train_split=0.7, val_split=0.15, early_stopping_patience=5, page_limit=100):
    def train():
        try:
            _reset_state()
            training_state['total_epochs'] = epochs
            training_state['status'] = 'preparing'
            training_state['message'] = 'Loading data...'
            model_path, _, plots_dir = _paths()
            primary = ["sentinel-2-l2a"]
            fallback = ["naip"]
            bbox = [-84.6, 33.7, -84.2, 34.1]
            date_range = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"
            try:
                X_all = core.build_preview_dataset(primary, bbox, date_range, page_limit=page_limit)
            except Exception:
                X_all = core.build_preview_dataset(fallback, bbox, date_range, page_limit=page_limit)
            training_state['message'] = 'Generating labels...'
            y_all, _, _ = core.haze_proxy_labels(X_all)
            rng = np.random.default_rng(42)
            idx = rng.permutation(len(X_all))
            n_train = int(train_split * len(idx))
            n_val = int(val_split * len(idx))
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]
            x_train = X_all[train_idx]
            y_train = y_all[train_idx]
            x_val = X_all[val_idx]
            y_val = y_all[val_idx]
            x_test = X_all[test_idx]
            y_test = y_all[test_idx]
            training_state['message'] = 'Building model...'
            model = core.build_model(
                input_shape=(*img_size, 3),
                num_classes=2,
                learning_rate=learning_rate,
                optimizer=optimizer,
                dropout_rate=dropout_rate,
                use_focal_loss=True,
            )
            train_ds = core.make_tf_ds(x_train, y_train, batch=batch_size, shuffle=True)
            val_ds = core.make_tf_ds(x_val, y_val, batch=batch_size, shuffle=False)
            ckpt_dir = os.path.join(plots_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, 'best_model.keras')
            callbacks = [
                ProgressCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=max(2, early_stopping_patience // 2),
                    min_lr=1e-7,
                    verbose=1,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    ckpt_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1,
                ),
            ]
            training_state['message'] = 'Training...'
            history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=0, callbacks=callbacks)
            if len(x_test) > 0:
                test_ds = core.make_tf_ds(x_test, y_test, batch=batch_size, shuffle=False)
                test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            else:
                test_loss, test_acc = model.evaluate(val_ds, verbose=0)
            training_state['final_stats']['test_accuracy'] = float(test_acc)
            training_state['final_stats']['test_loss'] = float(test_loss)
            h = history.history
            if h.get('accuracy'):
                training_state['final_stats']['train_accuracy'] = float(h['accuracy'][-1])
                training_state['final_stats']['train_loss'] = float(h['loss'][-1])
                training_state['final_stats']['val_accuracy'] = float(h['val_accuracy'][-1])
                training_state['final_stats']['val_loss'] = float(h['val_loss'][-1])
                training_state['final_stats']['total_epochs_trained'] = len(h['accuracy'])
            model.save(model_path)
            training_state['status'] = 'completed'
            training_state['message'] = 'Training completed successfully!'
            training_state['progress_percent'] = 100
        except Exception as e:
            import traceback
            training_state['status'] = 'error'
            training_state['message'] = str(e)
            training_state['error'] = traceback.format_exc()
            logger.exception("Training failed: %s", e)

    threading.Thread(target=train, daemon=True).start()
