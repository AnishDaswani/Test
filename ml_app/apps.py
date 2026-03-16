from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class MlAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_app'

    def ready(self):
        if __import__('os').environ.get('RUN_MAIN') != 'true':
            return
        try:
            __import__('importlib').import_module('tensorflow')
        except Exception:
            logger.warning("TensorFlow not available; skipping model init.")
            return
        import os
        import sys
        import threading
        from django.conf import settings

        def run_main():
            try:
                main_path = os.path.join(settings.BASE_DIR, 'main.py')
                if not os.path.exists(main_path):
                    return
                base = str(settings.BASE_DIR)
                if base not in sys.path:
                    sys.path.insert(0, base)
                import importlib.util
                spec = importlib.util.spec_from_file_location("main", main_path)
                if spec and getattr(spec, 'loader', None):
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                else:
                    import runpy
                    runpy.run_path(main_path, run_name="__main__")
            except Exception as e:
                logger.error("main.py init failed: %s", e)

        threading.Thread(target=run_main, daemon=True).start()
        logger.info("Pollution Detector app started.")
