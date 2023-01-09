from django.db import models

class Drinks(models.Model):
    name= models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return self.name


