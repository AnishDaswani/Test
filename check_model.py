import os
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except Exception as e:
    print("TensorFlow import failed:", e)
    raise
model_path='earthsearch_preview_haze_model.keras'
if os.path.exists(model_path):
    print('Found model file at', model_path)
    try:
        model = tf.keras.models.load_model(model_path)
        print('Model loaded. Summary:')
        model.summary()
    except Exception as e:
        print('Failed to load model:', e)
else:
    print('Model file not found at', model_path)
