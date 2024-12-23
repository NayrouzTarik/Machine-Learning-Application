from django.contrib import admin
from django.urls import path
from accounts import views as accounts_views
from MLApp import views as mlapp_views

urlpatterns = [
    path('', accounts_views.main_page, name='main_page'),
    path('inscription/', accounts_views.inscription, name='inscription'),
    path('connexion/', accounts_views.connexion, name='connexion'),
    path('deconnexion/', accounts_views.deconnexion, name='deconnexion'),
    path('acceuil/', accounts_views.acceuil, name='acceuil'),
    path('base/', accounts_views.base, name='base'),
    path('api/workflow-statistics', accounts_views.workflow_statistics, name='workflow-statistics'),
    path('upload/', mlapp_views.upload_data, name='upload_data'),
    path('clean_data/', mlapp_views.clean_data, name='clean_data'),
    path('run_model/', mlapp_views.run_model, name='run_model'),
    path('admin/', admin.site.urls),
]
