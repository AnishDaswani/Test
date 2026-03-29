from django.apps import AppConfig


class MlAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_app'

    def ready(self):
        from ml_app.startup import load_model
        load_model()
