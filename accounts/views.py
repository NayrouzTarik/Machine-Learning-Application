from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm 
from .form import CustomUserCreationForm
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect

from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json
from django.views.decorators.http import require_http_methods

def main_page(request):
    return render(request, 'main_page.html')
# Créer votre vue d'inscription
def inscription(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Votre compte a été créé avec succès. Vous pouvez maintenant vous connecter.')
            return redirect('connexion')
    else:
        form = CustomUserCreationForm()
    return render(request, 'inscription.html', {'form': form})

# Créer votre vue de connexion
def connexion(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        
        if user is not None:  # Vérifier si l'utilisateur est authentifié
            login(request, user)  # Connecter l'utilisateur
            
            # Vérifier si l'utilisateur est un administrateur
            if user.is_staff:
                return redirect('/admin/')  # Rediriger vers l'interface admin
            else:
                return redirect('base')  # Rediriger vers la page d'accueil des utilisateurs non-admin
        else:
            messages.error(request, 'Nom d\'utilisateur ou mot de passe incorrect.')
    
    return render(request, 'connexion.html')

# Créer la vue d'accueil pour les utilisateurs connectés
@login_required
def acceuil(request):
    return render(request, 'acceuil.html')


@login_required
def deconnexion(request):
    if request.method == 'POST':
        logout(request)
        return redirect('main_page')
    return redirect('main_page')
  
def base(request):
    return render(request, 'base.html')





@login_required
@require_http_methods(["POST"])
def update_email(request):
    try:
        data = json.loads(request.body)
        new_email = data.get('email')
        if new_email:
            request.user.email = new_email
            request.user.save()
            return JsonResponse({
                'success': True,
                'message': 'Email updated successfully!'
            })
        return JsonResponse({
            'success': False,
            'message': 'Invalid email provided.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)

@login_required
def workflow_statistics(request):
    stats = {
        'completed': 5,
        'in_progress': 2,
        'total': 7
    }
    return JsonResponse(stats)






@csrf_protect
def save_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Store the data in the session for use in later steps
            request.session['workflow_data'] = data['data']
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)