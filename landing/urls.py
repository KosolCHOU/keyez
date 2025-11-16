from django.urls import path
from .views import home, predict, transliterate

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict, name='predict'),
    path('transliterate/', transliterate, name='transliterate'),
]
