from django.db import models

class ImageUpload(models.Model):
    frame_image = models.ImageField(upload_to='images/')
    pool_image = models.ImageField(upload_to='images/')