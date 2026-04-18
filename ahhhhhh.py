import os
import json
import numpy as np
import tensorflow as tf

# Setup
model_path = 'models/earthsearch_preview_haze_model.keras'
out_dir = 'web/static/models'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)

# 1. Generate the Topology (JSON)
print("Building model.json...")
model_config = json.loads(model.to_json())

# 2. Extract and Flatten Weights
print("Packing binary weights...")
weights_data = []
weight_entries = []

for weight in model.weights:
    w_data = weight.numpy().astype('float32')
    w_bytes = w_data.tobytes()
    
    weight_entries.append({
        "name": weight.name,
        "shape": list(w_data.shape),
        "dtype": "float32"
    })
    weights_data.append(w_bytes)

# 3. Write the Binary Shard
bin_filename = "group1-shard1of1.bin"
with open(os.path.join(out_dir, bin_filename), 'wb') as f:
    for data in weights_data:
        f.write(data)

# 4. Finalize model.json
tfjs_manifest = {
    "modelTopology": model_config,
    "format": "layers-model",
    "generatedBy": "Manual-Script",
    "convertedBy": "The-Bypass-Method",
    "weightsManifest": [{
        "paths": [bin_filename],
        "weights": weight_entries
    }]
}

with open(os.path.join(out_dir, 'model.json'), 'w') as f:
    json.dump(tfjs_manifest, f)

print("🎉 ABSOLUTE SUCCESS! No libraries used, no dependencies crashed.")
print(f"Files created in: {out_dir}")