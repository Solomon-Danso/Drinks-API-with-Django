# Generated by Django 5.1.5 on 2025-02-03 18:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('snippets', '0003_roles'),
    ]

    operations = [
        migrations.AlterField(
            model_name='roles',
            name='userID',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
