from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('start/', views.start_game),
    path('move/', views.player_move),
    path('restart/', views.restart_game),
]