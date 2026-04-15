from django.db import models
from django.core.files.storage import default_storage
import os
import json

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    bbox = models.JSONField(default=list)
    date_range = models.CharField(max_length=255, default="2024-01-01T00:00:00Z/2024-12-31T23:59:59Z")
    collections = models.JSONField(default=list)
    image_count = models.IntegerField(default=0)
    status = models.CharField(max_length=50, default='pending')

    def __str__(self):
        return self.name

    def get_data_path(self):
        return f"datasets/{self.id}/"

class TrainingJob(models.Model):
    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default='pending')
    epochs = models.IntegerField(default=100)
    batch_size = models.IntegerField(default=32)
    learning_rate = models.FloatField(default=0.0005)
    config = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    model_path = models.CharField(max_length=500, blank=True)

    def __str__(self):
        return f"{self.name} - {self.status}"

class CustomImage(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='custom_images/')
    label = models.CharField(max_length=50, choices=[('clean', 'Clean'), ('polluted', 'Polluted')])
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.dataset.name} - {self.label}"