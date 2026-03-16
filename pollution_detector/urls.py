from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from ml_app import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index, name="index"),
    path("graphs/", views.graphs, name="graphs"),
    path("predict/", views.predict, name="predict"),
    path("predict/upload/", views.predict_upload, name="predict_upload"),
    path("train/", views.train_page, name="train"),
    path("train/start/", views.start_training, name="start_training"),
    path("train/progress/", views.get_training_progress, name="training_progress"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
