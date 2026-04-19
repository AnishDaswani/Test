from django.urls import path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('training/', views.training_dashboard, name='train'),
    path('gradcam/', views.gradcam_view, name='gradcam'),
    path('api/create-dataset/', views.create_dataset, name='create_dataset'),
    path('api/start-training/', views.start_training, name='start_training'),
    path('api/training-status/<int:job_id>/', views.get_training_status, name='training_status'),
    path('api/upload-custom-images/', views.upload_custom_images, name='upload_custom_images'),
    path('about/', TemplateView.as_view(template_name='about.html')),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
