from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_detected_letter/', views.get_detected_letter, name='get_detected_letter'),
    path('get_message/', views.get_message, name='get_message'),
    path('reset_message/', views.reset_message, name='reset_message'),
    path('start_message/', views.start_message, name='start_message'),
    path('stop_message/', views.stop_message, name='stop_message'),
]