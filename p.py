# AI, used to manually convert trained model into .json bc tensorflow is the problem, not me.
import os
import json
import numpy as np
import tensorflow as tf

model_path = 'models/earthsearch_preview_haze_model.keras'
out_dir = 'web/static/models'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)

def deep_clean(obj):
    if not isinstance(obj, dict): return obj
    if obj.get('class_name') == 'DTypePolicy': return 'float32'
    for key in ['module', 'registered_name', 'build_config']:
        obj.pop(key, None)
    for k, v in obj.items():
        if isinstance(v, dict): obj[k] = deep_clean(v)
        elif isinstance(v, list): obj[k] = [deep_clean(i) if isinstance(i, dict) else i for i in v]
    return obj

print("Building topology...")
full_config = json.loads(model.to_json())
all_layers = full_config['config']['layers'] if 'config' in full_config else full_config['layers']

bad_types = ['RandomFlip', 'RandomRotation', 'RandomZoom', 'RandomContrast', 'Rescaling', 'Normalization']
cleaned_layers = []

# Standard InputLayer
cleaned_layers.append({
    "class_name": "InputLayer",
    "config": {"batch_input_shape": [None, 128, 128, 3], "dtype": "float32", "sparse": False, "name": "input_1"}
})

print("Syncing weights and cleaning layers...")
weight_entries = []
weights_data = []

for layer in model.layers:
    if layer.__class__.__name__ in bad_types:
        continue
    
    # 1. Add cleaned layer to topology
    for l_conf in all_layers:
        if l_conf['config'].get('name') == layer.name:
            cleaned_layers.append(deep_clean(l_conf))
            break
    
    # 2. Add weights with clean names (strip ALL prefixes)
    for weight in layer.weights:
        w_data = weight.numpy().astype('float32')
        
        # This forces the name to be exactly 'layer_name/weight_type' 
        # Example: 'conv2d/kernel'
        weight_type = weight.name.split('/')[-1].replace(':0', '')
        unique_name = f"{layer.name}/{weight_type}"
        
        weight_entries.append({
            "name": unique_name,
            "shape": list(w_data.shape),
            "dtype": "float32"
        })
        weights_data.append(w_data.tobytes())

# 3. Save Files
with open(os.path.join(out_dir, "group1-shard1of1.bin"), 'wb') as f:
    for data in weights_data:
        f.write(data)

with open(os.path.join(out_dir, 'model.json'), 'w') as f:
    json.dump({
        "modelTopology": {"class_name": "Sequential", "config": {"name": "sequential", "layers": cleaned_layers}},
        "format": "layers-model",
        "generatedBy": "Final-Force-Sync",
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": weight_entries
        }]
    }, f, indent=2)

print(f"🎉 SYNC SUCCESS! Saved to: {out_dir}")
