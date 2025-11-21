from django.urls import path
from .views import home, predict, transliterate, grammar_checker

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict, name='predict'),
    path('transliterate/', transliterate, name='transliterate'),
    path('grammar-checker/', grammar_checker, name='grammar_checker'),
]
