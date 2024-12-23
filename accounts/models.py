from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Vous pouvez ajouter des champs personnalisés ici
    # Par exemple :
    phone_number = models.CharField(max_length=20, blank=True)

    class Meta:
        verbose_name = 'user'
        verbose_name_plural = 'users'

    # Modifiez les relations pour éviter les conflits de reverse accessors
    groups = models.ManyToManyField(
        'auth.Group',
        related_name='customuser_set',  # Utilisez un nom unique pour l'accès inverse
        blank=True,
        help_text='The groups this user belongs to.',
        related_query_name='customuser'
    )
    
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='customuser_permissions',  # Utilisez un nom unique pour l'accès inverse
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='customuser'
    )








