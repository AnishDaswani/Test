import os
import sys
import django
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / 'web'
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_detector.settings')
django.setup()

from django.core.management import execute_from_command_line

if __name__ == '__main__':
    execute_from_command_line(['manage.py', 'runserver', '8000'])