import os
import subprocess
import sys

model_path = 'models/earthsearch_preview_haze_model.keras'
output_dir = 'web/static/models'

os.makedirs(output_dir, exist_ok=True)

if os.path.exists(model_path):
    try:
        subprocess.run([
            sys.executable, '-m', 'tensorflowjs_converter',
            '--input_format=keras',
            model_path,
            output_dir
        ], check=True)
        print(f"Model converted and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        print("Make sure tensorflowjs is installed: pip install tensorflowjs")
else:
    print(f"Model not found at {model_path}")