import os
import sys
import django
from pathlib import Path

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    WEB_DIR = REPO_ROOT / 'web'
    
    sys.path.insert(0, str(WEB_DIR))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_detector.settings')
    
    django.setup()
    from django.core.management import execute_from_command_line
    execute_from_command_line([sys.argv[0], 'runserver', '0.0.0.0:8000'])

if __name__ == '__main__':
    main()
