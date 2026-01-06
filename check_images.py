import os, glob
files = glob.glob('plots/**/*.png', recursive=True) + glob.glob('*.png')
if not files:
    print('No PNG files found')
else:
    for f in sorted(files):
        try:
            print(f, os.path.getsize(f))
        except Exception as e:
            print(f, 'ERROR', e)
