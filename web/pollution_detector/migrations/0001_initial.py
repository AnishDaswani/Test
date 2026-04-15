import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('bbox', models.JSONField(default=list)),
                ('date_range', models.CharField(default='2024-01-01T00:00:00Z/2024-12-31T23:59:59Z', max_length=255)),
                ('collections', models.JSONField(default=list)),
                ('image_count', models.IntegerField(default=0)),
                ('status', models.CharField(default='pending', max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='CustomImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='custom_images/')),
                ('label', models.CharField(choices=[('clean', 'Clean'), ('polluted', 'Polluted')], max_length=50)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='pollution_detector.dataset')),
            ],
        ),
        migrations.CreateModel(
            name='TrainingJob',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(default='pending', max_length=50)),
                ('epochs', models.IntegerField(default=100)),
                ('batch_size', models.IntegerField(default=32)),
                ('learning_rate', models.FloatField(default=0.0005)),
                ('config', models.JSONField(default=dict)),
                ('metrics', models.JSONField(default=dict)),
                ('model_path', models.CharField(blank=True, max_length=500)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='pollution_detector.dataset')),
            ],
        ),
    ]
